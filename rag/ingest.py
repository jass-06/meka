"""
rag/ingest.py

Extracts text from uploaded/discovered files (PDF, DOCX, TXT, Markdown) and
splits it into overlapping chunks ready for embedding.
"""
import io
import re
import uuid
from typing import Dict, List, Tuple

import pdfplumber
from docx import Document as DocxDocument

from rag.config import Config


class DocumentIngester:
    def __init__(self, chunk_size: int = None, chunk_overlap: int = None):
        self.chunk_size = chunk_size or Config.CHUNK_SIZE
        self.chunk_overlap = chunk_overlap or Config.CHUNK_OVERLAP

    # --- extraction ---------------------------------------------------------

    def extract_text(self, file_bytes: bytes, filename: str) -> str:
        ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

        if ext == "pdf":
            return self._extract_pdf(file_bytes)
        if ext in ("docx", "doc"):
            return self._extract_docx(file_bytes)
        if ext in ("txt", "md", "markdown"):
            return file_bytes.decode("utf-8", errors="ignore")
        raise ValueError(f"Unsupported file type: .{ext}")

    @staticmethod
    def _extract_pdf(file_bytes: bytes) -> str:
        text_parts = []
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            for i, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(f"[page {i}]\n{page_text}")
        return "\n\n".join(text_parts)

    @staticmethod
    def _extract_docx(file_bytes: bytes) -> str:
        doc = DocxDocument(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    # --- chunking -------------------------------------------------------------

    def chunk_text(self, text: str) -> List[str]:
        text = re.sub(r"\n{3,}", "\n\n", text.strip())
        if not text:
            return []

        chunks = []
        start = 0
        n = len(text)
        while start < n:
            end = min(start + self.chunk_size, n)
            # try to break on a sentence/paragraph boundary near the end
            if end < n:
                boundary = text.rfind("\n", start, end)
                if boundary == -1 or boundary <= start:
                    boundary = text.rfind(". ", start, end)
                if boundary > start:
                    end = boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= n:
                break
            start = max(end - self.chunk_overlap, start + 1)
        return chunks

    # --- top-level processing ---------------------------------------------------

    def process_file(self, file_bytes: bytes, filename: str) -> List[Dict]:
        """Return a list of {id, text, metadata} chunk records for one file."""
        text = self.extract_text(file_bytes, filename)
        chunks = self.chunk_text(text)

        records = []
        for i, chunk in enumerate(chunks):
            records.append(
                {
                    "id": f"{filename}-{i}-{uuid.uuid4().hex[:8]}",
                    "text": chunk,
                    "metadata": {"source": filename, "chunk_index": i},
                }
            )
        return records

    def process_files(self, files: List[Tuple[bytes, str]]) -> List[Dict]:
        """files: list of (file_bytes, filename). Skips files that fail to parse."""
        all_records = []
        errors = []
        for file_bytes, filename in files:
            try:
                all_records.extend(self.process_file(file_bytes, filename))
            except Exception as e:
                errors.append(f"{filename}: {e}")
        if errors:
            print(f"[ingest] {len(errors)} file(s) failed: {errors}")
        return all_records
