"""
rag/vector_store.py

Wraps a persistent ChromaDB collection. Embeddings are computed elsewhere
(rag.embedder) and passed in directly, so this class never talks to HF.
"""
from typing import Dict, List, Optional

import chromadb

from rag.config import Config


class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(path=Config.CHROMA_PATH)
        self.collection = self.client.get_or_create_collection(
            name=Config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict],
    ) -> None:
        if not ids:
            return
        self.collection.add(
            ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas
        )

    def query(self, embedding: List[float], n_results: int) -> Dict:
        count = self.collection.count()
        if count == 0:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]], "ids": [[]]}
        return self.collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results, count),
            include=["documents", "metadatas", "distances"],
        )

    def count(self) -> int:
        return self.collection.count()

    def is_empty(self) -> bool:
        return self.count() == 0

    def clear(self) -> bool:
        try:
            self.client.delete_collection(name=Config.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=Config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
            )
            return True
        except Exception:
            return False

    def get_stats(self) -> Dict:
        try:
            return {
                "document_count": self.count(),
                "persist_directory": Config.CHROMA_PATH,
                "collection_name": Config.COLLECTION_NAME,
            }
        except Exception as e:
            return {"error": str(e)}
