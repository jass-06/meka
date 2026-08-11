# MEKA — Multimodal Enterprise Knowledge Assistant

A RAG (Retrieval-Augmented Generation) system for chatting with internal
company documents — policies, security guides, HR docs — with optional
image input. Built API-first: the same pipeline powers a Streamlit UI, a
FastAPI REST service, and a terminal chat client.

## Why this exists

Most take-home RAG demos are a single Streamlit script glued to one LLM
call. MEKA is structured like something you'd actually deploy:

- **API-first** — `rag/retrieval.py` exposes one `RAGPipeline` class; the
  Streamlit UI, the FastAPI service, the CLI, and `evaluation.py` are all
  thin clients of it. No logic is duplicated between them.
- **Dual LLM routing** — Ollama (local, offline, free) is the default;
  Groq (cloud) is used as an on-demand "assist" mode, with automatic
  fallback either direction if a backend is down. Optional HyDE
  (hypothetical-document-embedding) query expansion improves recall in
  assist mode.
- **Multimodal** — images are captioned via a HuggingFace vision model and
  folded into the same text-embedding retrieval path, so you don't need a
  separate image vector index.
- **Reranked retrieval** — a lightweight reranker blends vector distance
  with keyword overlap so short, term-specific queries (policy names,
  error codes) don't get lost to purely semantic drift.
- **Tested, linted, containerized, CI'd** — see below.

## Architecture

```
meka/
├── app.py                  # Streamlit chat UI
├── api.py                  # FastAPI REST service (same pipeline, headless)
├── meka_chat.py             # Terminal chat client
├── offline_ingest.py        # Batch-ingest everything in ./data
├── evaluation.py            # Retrieval + answer-quality eval harness
├── evaluation_data.json     # Sample eval questions/answers
├── rag/
│   ├── config.py            # All env-driven settings
│   ├── ingest.py             # Text extraction (PDF/DOCX/TXT/MD) + chunking
│   ├── embedder.py           # HuggingFace embedding client
│   ├── vision.py              # HuggingFace image captioning client
│   ├── vector_store.py        # ChromaDB wrapper
│   ├── reranker.py             # Semantic + keyword blended reranking
│   ├── llm.py                   # Groq / Ollama routing + HyDE
│   └── retrieval.py              # RAGPipeline — ties it all together
├── tests/                    # Offline unit tests (no external API calls)
├── data/                      # Seed documents ingested by offline_ingest.py
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci.yml   # Lint + test on every push/PR
```

### How a question gets answered

1. `RAGPipeline.process_question()` optionally captions an attached image
   and (in assist mode) generates a HyDE hypothetical answer.
2. The (possibly expanded) query is embedded and used to pull the top
   candidates from ChromaDB.
3. `Reranker` re-scores those candidates by blending vector distance with
   keyword overlap, returning the top K.
4. `LLMManager` builds a context block from those chunks and calls
   Ollama or Groq (with fallback), returning the answer plus which
   backend answered and why.

## Setup

### 1. Clone and install

```bash
git clone https://github.com/jass-06/meka.git
cd meka
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:
- `HF_API_TOKEN` — **required**. Get one free at https://huggingface.co/settings/tokens
- `GROQ_API_KEY` — optional, enables cloud assist mode. Get one at https://console.groq.com/keys
- `MEKA_API_KEY` — optional, protects the REST API. Set it before deploying anywhere public.

### 3. (Optional) Set up Ollama for local/offline answering

```bash
# https://ollama.ai
ollama pull llama3.2
ollama serve
```

If you skip this, MEKA still works as long as `GROQ_API_KEY` is set — it
will just always use Groq.

### 4. Ingest documents

Two sample policy docs are already in `data/` so you can try it
immediately. Add your own `.txt`, `.md`, `.pdf`, or `.docx` files there,
then:

```bash
python offline_ingest.py
```

### 5. Run it

**Streamlit UI:**
```bash
streamlit run app.py
```
Open http://localhost:8501

**REST API:**
```bash
uvicorn api:app --reload --port 8000
```
Docs at http://localhost:8000/docs

**Terminal:**
```bash
python meka_chat.py
```

## REST API

All routes except `/health` require header `X-API-Key: <MEKA_API_KEY>` if
you've set one in `.env`.

| Method | Route | Description |
|---|---|---|
| GET | `/health` | Liveness check (no auth) |
| GET | `/stats` | Vector store + LLM backend status |
| POST | `/query` | `{"question": "...", "use_groq": false, "image_base64": null}` → answer |
| POST | `/ingest` | Multipart file upload → ingest into the vector store |
| DELETE | `/documents` | Clear the vector store |

Example:
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: $MEKA_API_KEY" \
  -d '{"question": "What is the VPN policy for remote work?"}'
```

## Docker

```bash
docker compose up --build
```

This starts both the UI (`:8501`) and the API (`:8000`), sharing a
`chroma_data` volume. Put your `.env` next to `docker-compose.yml` first.

> If you're running Ollama on your host machine (not in Docker), set
> `OLLAMA_URL=http://host.docker.internal:11434/api/chat` in `.env` so the
> containers can reach it.

## Testing & CI

```bash
pytest tests/ -v
flake8 rag api.py app.py
```

`.github/workflows/ci.yml` runs both on every push/PR to `main`. The unit
tests only exercise pure logic (chunking, reranking, the API's request
validation) so CI needs no secrets or live services.

`evaluation.py` is a separate, heavier harness that actually calls the
live pipeline (embeddings + LLM) against `evaluation_data.json` and
reports retrieval precision/recall and answer EM/F1:
```bash
python evaluation.py --k 5
```

## Known limitations

- Scanned/image-only PDFs won't extract text (no OCR yet).
- The reranker is lexical + vector-distance, not a trained cross-encoder —
  good enough for demo-scale corpora, not a production-grade reranker.
- `MEKA_API_KEY` is a single shared secret, not per-user auth — fine for a
  small team/demo, not for multi-tenant production use.

## License

MIT — see [LICENSE](LICENSE).
