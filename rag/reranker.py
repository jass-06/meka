"""
rag/reranker.py

A dependency-light reranker: it re-scores the top vector-search candidates
by blending semantic distance with lexical (keyword) overlap against the
query. This catches cases where semantic search alone drifts off-topic on
short, term-specific enterprise queries (policy names, error codes, etc.)
without requiring a heavyweight cross-encoder model.
"""
import re
from typing import Dict, List


def _tokenize(text: str) -> set:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


class Reranker:
    def __init__(self, keyword_weight: float = 0.35):
        self.keyword_weight = keyword_weight

    def rerank(self, query: str, candidates: List[Dict], top_k: int) -> List[Dict]:
        """
        candidates: list of {"text": str, "metadata": dict, "distance": float}
        Lower distance = more similar. Returns top_k candidates sorted best-first.
        """
        if not candidates:
            return []

        query_tokens = _tokenize(query)
        max_distance = max((c["distance"] for c in candidates), default=1.0) or 1.0

        scored = []
        for c in candidates:
            # semantic score: closer distance -> higher score, normalized to [0, 1]
            semantic_score = 1.0 - (c["distance"] / max_distance)

            doc_tokens = _tokenize(c["text"])
            overlap = len(query_tokens & doc_tokens)
            keyword_score = overlap / max(len(query_tokens), 1)

            final_score = (1 - self.keyword_weight) * semantic_score + self.keyword_weight * keyword_score
            scored.append({**c, "score": final_score})

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]
