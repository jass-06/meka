"""
Offline ingestion script for TechNova Enterprise Knowledge Assistant.

Reads all supported files from ./data, processes + chunks them via
DocumentIngester (including advanced PDF processing), generates embeddings,
and stores them in Chroma.
"""

import os
import time

from rag.retrieval import RAGPipeline
from rag.config import Config

# Supported file types for ingestion
SUPPORTED_EXTS = {".txt", ".md", ".markdown", ".pdf", ".docx", ".doc"}


def discover_files(data_dir: str = "data"):
    """Find all supported files in the data directory and load them as bytes."""
    files = []

    for name in os.listdir(data_dir):
        path = os.path.join(data_dir, name)
        if not os.path.isfile(path):
            continue

        _, ext = os.path.splitext(name)
        if ext.lower() not in SUPPORTED_EXTS:
            continue

        try:
            with open(path, "rb") as f:
                content = f.read()
            files.append((content, name))
        except Exception as e:
            print(f"[WARN] Skipping {name} (failed to read: {e})")

    return files


def main():
    print("🚀 Starting offline document ingestion...")

    # If Config has DATA_DIR use it, otherwise default to "data"
    data_dir = getattr(Config, "DATA_DIR", "data")

    if not os.path.isdir(data_dir):
        raise SystemExit(f"[ERROR] Data directory not found: {data_dir}")

    files = discover_files(data_dir)
    if not files:
        print("[INFO] No supported files found in data directory.")
        return

    print(f"[INFO] Found {len(files)} files to process:")
    for _, name in files:
        print(f"  - {name}")

    # Initialize full RAG pipeline (embedder + vector store + LLM manager)
    pipeline = RAGPipeline()

    # Optional: clear existing vector store so we start clean
    print("\n[INFO] Clearing existing vector store collection...")
    cleared = pipeline.clear_documents()
    if cleared:
        print("[INFO] Existing collection cleared.")
    else:
        print("[WARN] Could not clear collection (may be empty or error occurred).")

    # Ingest & embed
    start = time.time()
    result = pipeline.ingest_documents(files)
    elapsed = time.time() - start

    print("\n📊 Ingestion summary")
    print(f"  Success:          {result.get('success')}")
    print(f"  Message:          {result.get('message')}")
    print(f"  Chunks processed: {result.get('chunks_processed')}")
    print(f"  Total chunks:     {result.get('total_chunks')}")
    print(f"  Time taken:       {elapsed:.1f}s")

    # Show vector store stats
    stats = pipeline.get_stats().get("vector_store", {})
    if "error" not in stats:
        print(f"\nVector store now has {stats.get('document_count', 0)} documents.")
        print(f"Persist directory: {stats.get('persist_directory')}")
    else:
        print(f"\n[WARN] Could not read vector store stats: {stats['error']}")


if __name__ == "__main__":
    main()