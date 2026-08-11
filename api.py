"""
api.py

FastAPI REST layer around the RAG pipeline. Lets MEKA be used headlessly
(curl, another service, a frontend other than Streamlit) instead of only
through the Streamlit UI.

Run with:
    uvicorn api:app --reload --port 8000

Auth: every route except /health requires header `X-API-Key: <MEKA_API_KEY>`
if MEKA_API_KEY is set in the environment. Leave it unset for local dev.
"""
from typing import List, Optional

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from pydantic import BaseModel

from rag.config import Config
from rag.retrieval import RAGPipeline

app = FastAPI(
    title="MEKA API",
    description="Multimodal Enterprise Knowledge Assistant - REST API",
    version="1.0.0",
)

pipeline = RAGPipeline()


def require_api_key(x_api_key: Optional[str] = Header(default=None)):
    if Config.MEKA_API_KEY and x_api_key != Config.MEKA_API_KEY:
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key header.")
    return True


class QueryRequest(BaseModel):
    question: str
    image_base64: Optional[str] = None
    use_groq: bool = False


class QueryResponse(BaseModel):
    answer: str
    sources: List[str] = []
    used_llm: str = "none"
    router_reason: str = ""
    error: Optional[str] = None


class IngestResponse(BaseModel):
    success: bool
    message: str
    chunks_processed: int = 0
    total_chunks: Optional[int] = None


@app.get("/health")
def health():
    """Unauthenticated liveness check, useful for Docker/CI."""
    return {"status": "ok"}


@app.get("/stats", dependencies=[Depends(require_api_key)])
def stats():
    return pipeline.get_stats()


@app.post("/query", response_model=QueryResponse, dependencies=[Depends(require_api_key)])
def query(req: QueryRequest):
    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="`question` must not be empty.")
    result = pipeline.process_question(
        question=req.question, image_base64=req.image_base64, use_groq=req.use_groq
    )
    return result


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(files: List[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")
    payload = [(await f.read(), f.filename) for f in files]
    result = pipeline.ingest_documents(payload)
    return result


@app.delete("/documents", dependencies=[Depends(require_api_key)])
def clear_documents():
    ok = pipeline.clear_documents()
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to clear documents.")
    return {"success": True}
