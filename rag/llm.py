"""
rag/llm.py

Manages the two LLM backends:
  - Ollama (local, default, fully offline once pulled)
  - Groq (cloud, used as "assist" when explicitly requested or when Ollama
    is unavailable / retrieval looks weak)

Also provides a small HyDE (Hypothetical Document Embeddings) helper: when
use_groq is enabled, we ask the LLM to sketch a hypothetical answer first
and embed *that* for retrieval, which tends to improve recall on short or
vaguely-worded questions.
"""
from typing import Optional, Tuple

import requests

from rag.config import Config

SYSTEM_PROMPT = (
    "You are MEKA, an enterprise knowledge assistant for internal company "
    "policies, security, legal, HR, and IT documentation. "
    "Answer ONLY using the provided context. Do not combine unrelated policies "
    "and do not infer beyond the retrieved text. "
    "If the answer is not clearly supported by the context, say you don't know "
    "rather than guessing. "
    "When you do answer, briefly reference which source(s) you used."
)


class LLMManager:
    def __init__(self):
        self.groq_key = Config.GROQ_API_KEY
        self.ollama_url = Config.OLLAMA_URL
        self.ollama_model = Config.OLLAMA_MODEL
        self.timeout = Config.REQUEST_TIMEOUT

    # --- availability checks -------------------------------------------------

    def groq_available(self) -> bool:
        return bool(self.groq_key)

    def ollama_available(self) -> bool:
        try:
            base = self.ollama_url.split("/api/")[0]
            resp = requests.get(f"{base}/api/tags", timeout=3)
            return resp.status_code == 200
        except Exception:
            return False

    # --- backend calls -----------------------------------------------------------

    def _call_groq(self, user_content: str) -> str:
        resp = requests.post(
            Config.GROQ_API_URL,
            headers={
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": Config.GROQ_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["choices"][0]["message"]["content"]

    def _call_ollama(self, user_content: str) -> str:
        resp = requests.post(
            self.ollama_url,
            json={
                "model": self.ollama_model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                "stream": False,
            },
            timeout=self.timeout,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    # --- public API --------------------------------------------------------------

    def generate(self, question: str, context: str, use_groq: bool = False) -> Tuple[str, str, str]:
        """
        Returns (answer, used_llm, router_reason).
        Routing: if use_groq is requested and Groq is configured, try Groq first,
        falling back to Ollama on failure. Otherwise try Ollama first, falling
        back to Groq only if Ollama is unavailable and Groq is configured.
        """
        user_content = f"Context:\n{context or '(no relevant context found)'}\n\nQuestion:\n{question}"

        order = []
        if use_groq and self.groq_available():
            order = [("groq", "requested by user"), ("ollama", "groq call failed, falling back")]
        else:
            order = [("ollama", "default local backend")]
            if self.groq_available():
                order.append(("groq", "ollama unavailable, using cloud assist"))

        last_error = None
        for backend, reason in order:
            try:
                if backend == "groq":
                    answer = self._call_groq(user_content)
                else:
                    answer = self._call_ollama(user_content)
                return answer, backend, reason
            except Exception as e:
                last_error = e
                continue

        return (
            f"I couldn't reach any configured LLM backend. Last error: {last_error}",
            "none",
            "all backends failed",
        )

    def generate_hyde(self, question: str) -> Optional[str]:
        """Generate a short hypothetical answer to embed for HyDE retrieval."""
        prompt = (
            "Write a brief (2-3 sentence) hypothetical answer to this enterprise "
            f"knowledge-base question, even if you are not certain it's correct. "
            f"Question: {question}"
        )
        try:
            if self.groq_available():
                return self._call_groq(prompt)
            if self.ollama_available():
                return self._call_ollama(prompt)
        except Exception:
            return None
        return None
