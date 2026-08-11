"""
rag/embedder.py

Thin wrapper around the HuggingFace Inference (router) API for turning text
into embedding vectors, with basic retry handling and batching.
"""
import time
from typing import List

import requests

from rag.config import Config


class EmbeddingError(RuntimeError):
    pass


class HFEmbedder:
    def __init__(self):
        self.url = Config.HF_EMBEDDING_URL
        self.token = Config.HF_API_TOKEN
        self.timeout = Config.REQUEST_TIMEOUT

    def is_available(self) -> bool:
        return bool(self.token)

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
        }

    def _post(self, payload: dict, retries: int = 3, backoff: float = 1.5):
        last_err = None
        for attempt in range(retries):
            try:
                resp = requests.post(
                    self.url, headers=self._headers(), json=payload, timeout=self.timeout
                )
                if resp.status_code == 503:
                    # Model is warming up on HF's side - wait and retry
                    time.sleep(backoff * (attempt + 1))
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                last_err = e
                time.sleep(backoff * (attempt + 1))
        raise EmbeddingError(f"HF embedding request failed after {retries} attempts: {last_err}")

    @staticmethod
    def _normalize(data) -> List[float]:
        # Router can return a flat vector, a batch of one, or nested pooled output
        if isinstance(data, list) and data and isinstance(data[0], (int, float)):
            return data
        if isinstance(data, list) and data and isinstance(data[0], list):
            first = data[0]
            if first and isinstance(first[0], (int, float)):
                return first
            # token-level embeddings [[[...]], ...] -> mean pool
            flat = [tok for tok in first]
            return [sum(vals) / len(vals) for vals in zip(*flat)]
        raise EmbeddingError(f"Unexpected embedding response shape: {type(data)}")

    def embed_text(self, text: str) -> List[float]:
        if not self.is_available():
            raise EmbeddingError("HF_API_TOKEN is not configured.")
        data = self._post({"inputs": text})
        return self._normalize(data)

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if not self.is_available():
            raise EmbeddingError("HF_API_TOKEN is not configured.")
        # The router endpoint accepts a list under "inputs" for batched calls.
        data = self._post({"inputs": texts})
        if isinstance(data, list) and data and isinstance(data[0], list) and isinstance(data[0][0], (int, float)):
            return data
        # Fall back to one-by-one if the batch response shape is unexpected
        return [self.embed_text(t) for t in texts]
