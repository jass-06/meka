"""
rag/retrieval.py

Assembles ingestion, embedding, vector search, reranking, and LLM generation
into a single RAGPipeline class. This is the one object the UI (app.py),
the REST API (api.py), the offline ingestion script, and evaluation.py all
depend on - keep its public method signatures stable.
"""
from typing import Any, Dict, List, Optional, Tuple

from rag.config import Config
from rag.embedder import HFEmbedder, EmbeddingError
from rag.ingest import DocumentIngester
from rag.llm import LLMManager
from rag.reranker import Reranker
from rag.vector_store import VectorStore
from rag.vision import ImageCaptioner


class RAGPipeline:
    def __init__(self):
        self.embedder = HFEmbedder()
        self.vector_store = VectorStore()
        self.ingester = DocumentIngester()
        self.llm = LLMManager()
        self.reranker = Reranker()
        self.captioner = ImageCaptioner()

    # --- ingestion -----------------------------------------------------------------

    def ingest_documents(self, files: List[Tuple[bytes, str]]) -> Dict[str, Any]:
        """files: list of (file_bytes, filename)."""
        try:
            records = self.ingester.process_files(files)
            if not records:
                return {
                    "success": False,
                    "message": "No text could be extracted from the provided files.",
                    "chunks_processed": 0,
                    "total_chunks": self.vector_store.count(),
                }

            texts = [r["text"] for r in records]
            embeddings = self.embedder.embed_batch(texts)

            self.vector_store.add(
                ids=[r["id"] for r in records],
                embeddings=embeddings,
                documents=texts,
                metadatas=[r["metadata"] for r in records],
            )

            return {
                "success": True,
                "message": f"Ingested {len(files)} file(s) into {len(records)} chunks.",
                "chunks_processed": len(records),
                "total_chunks": self.vector_store.count(),
            }
        except EmbeddingError as e:
            return {"success": False, "message": f"Embedding error: {e}", "chunks_processed": 0}
        except Exception as e:
            return {"success": False, "message": f"Ingestion failed: {e}", "chunks_processed": 0}

    def clear_documents(self) -> bool:
        return self.vector_store.clear()

    def is_empty(self) -> bool:
        return self.vector_store.is_empty()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "vector_store": self.vector_store.get_stats(),
            "llm_backends": {
                "groq": self.llm.groq_available(),
                "ollama": self.llm.ollama_available(),
            },
        }

    # --- retrieval + generation -----------------------------------------------------

    def _retrieve(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        query_embedding = self.embedder.embed_text(query_text)
        raw = self.vector_store.query(query_embedding, n_results=Config.RETRIEVAL_CANDIDATES)

        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]

        candidates = [
            {"text": doc, "metadata": meta, "distance": dist}
            for doc, meta, dist in zip(docs, metas, dists)
        ]
        return self.reranker.rerank(query_text, candidates, top_k=top_k)

    def process_question(
        self,
        question: str,
        image_base64: Optional[str] = None,
        use_groq: bool = False,
    ) -> Dict[str, Any]:
        try:
            if self.is_empty():
                return {
                    "answer": "I don't have any documents ingested yet. Please add files to "
                    "`data/` and run `python offline_ingest.py` first.",
                    "sources": [],
                    "used_llm": "none",
                    "router_reason": "empty vector store",
                }

            # Fold an optional image caption into the retrieval query.
            retrieval_query = question
            if image_base64:
                caption = self.captioner.caption(image_base64)
                if caption:
                    retrieval_query = f"{question}\n(Attached image shows: {caption})"

            # Optional HyDE: when Groq assist is on, retrieve using a hypothetical
            # answer as well as the raw question, to improve recall.
            if use_groq:
                hyde = self.llm.generate_hyde(retrieval_query)
                if hyde:
                    retrieval_query = f"{retrieval_query}\n{hyde}"

            top_chunks = self._retrieve(retrieval_query, top_k=Config.TOP_K)

            context = "\n\n---\n\n".join(
                f"[Source: {c['metadata'].get('source', 'unknown')}]\n{c['text']}"
                for c in top_chunks
            )

            answer, used_llm, router_reason = self.llm.generate(
                question=question, context=context, use_groq=use_groq
            )

            sources = sorted({c["metadata"].get("source", "unknown") for c in top_chunks})

            return {
                "answer": answer,
                "sources": sources,
                "used_llm": used_llm,
                "router_reason": router_reason,
            }
        except EmbeddingError as e:
            return {"answer": "", "sources": [], "used_llm": "none", "router_reason": "", "error": str(e)}
        except Exception as e:
            return {"answer": "", "sources": [], "used_llm": "none", "router_reason": "", "error": str(e)}
