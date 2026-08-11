"""
rag/config.py

Central configuration for MEKA. All environment-driven settings live here so
the rest of the codebase never touches os.environ directly.
"""
import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # --- API keys / tokens -------------------------------------------------
    HF_API_TOKEN: str = os.getenv("HF_API_TOKEN", "")
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")

    # API key required to call the FastAPI service (see api.py). Leave empty
    # in .env to disable auth for local dev, but ALWAYS set this in prod.
    MEKA_API_KEY: str = os.getenv("MEKA_API_KEY", "")

    # --- Embedding model -----------------------------------------------------
    HF_EMBEDDING_MODEL: str = os.getenv(
        "HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    HF_EMBEDDING_URL: str = (
        f"https://router.huggingface.co/hf-inference/models/"
        f"{HF_EMBEDDING_MODEL}/pipeline/feature-extraction"
    )

    # --- Vision / captioning model --------------------------------------------
    HF_VISION_MODEL: str = os.getenv(
        "HF_VISION_MODEL", "Salesforce/blip-image-captioning-base"
    )
    HF_VISION_URL: str = f"https://router.huggingface.co/hf-inference/models/{HF_VISION_MODEL}"

    # --- Vector store ----------------------------------------------------------
    CHROMA_PATH: str = os.getenv("CHROMA_PATH", "./chroma_db")
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "enterprise_knowledge")

    # --- Chunking ----------------------------------------------------------------
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "150"))

    # --- LLM backends --------------------------------------------------------------
    GROQ_MODEL: str = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
    GROQ_API_URL: str = "https://api.groq.com/openai/v1/chat/completions"

    OLLAMA_URL: str = os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat")
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3.2")

    # --- Retrieval -------------------------------------------------------------------
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    RETRIEVAL_CANDIDATES: int = int(os.getenv("RETRIEVAL_CANDIDATES", "15"))  # pre-rerank pool

    # --- Ingestion ---------------------------------------------------------------------
    DATA_DIR: str = os.getenv("DATA_DIR", "data")

    # --- Misc -----------------------------------------------------------------------------
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "60"))

    @classmethod
    def validate_config(cls) -> bool:
        """Return True only if the minimum required config is present."""
        return bool(cls.HF_API_TOKEN)

    @classmethod
    def backend_summary(cls) -> dict:
        return {
            "hf_token_set": bool(cls.HF_API_TOKEN),
            "groq_key_set": bool(cls.GROQ_API_KEY),
            "ollama_url": cls.OLLAMA_URL,
            "api_auth_enabled": bool(cls.MEKA_API_KEY),
        }
