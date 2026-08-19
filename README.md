# Enterprise RAG Platform

A production-shaped Retrieval-Augmented Generation platform for chatting with company documents — PDF, DOCX, TXT, Markdown, and images (via AI captioning). Ships with **both** a Streamlit chat UI and a **FastAPI REST API** behind API-key auth, a Dockerized deployment, and a CI pipeline that runs the test suite on every push.

This is a from-scratch rebuild of an earlier prototype, restructured to close the gaps a prototype has vs. a project you'd actually put in production: an API layer, containerization, automated tests in CI, access control, and a clean repo (no build artifacts committed).

## Architecture

```
enterprise-rag-platform/
├── app.py                  # Streamlit chat UI
├── api/
│   ├── main.py              # FastAPI app: /health /ingest /chat /reset
│   ├── auth.py               # X-API-Key header gating
│   └── schemas.py            # Request/response models
├── rag/
│   ├── config.py              # All env-driven settings, one place
│   ├── ingest.py               # PDF/DOCX/TXT/MD/image -> text -> chunks
│   ├── embedder.py              # sentence-transformers wrapper
│   ├── vision.py                 # HuggingFace image captioning
│   ├── vector_store.py            # ChromaDB persistent collection
│   ├── llm.py                      # Groq primary, Ollama fallback
│   └── retrieval.py                 # Wires embed -> retrieve -> generate
├── tests/                    # pytest suite (mocked, no API keys needed in CI)
├── .github/workflows/test.yml  # CI: installs deps, runs pytest on every push/PR
├── Dockerfile
├── docker-compose.yml          # Runs API (8000) + UI (8501) together
├── .env.example
└── .gitignore                    # data/ and chroma_db/ are NOT committed
```

Both the UI and the API call the exact same `rag/retrieval.py` pipeline, so behavior never drifts between the two entry points.

## Features

- **Multi-format ingestion**: PDF, DOCX, TXT, Markdown, PNG/JPG (captioned via HuggingFace vision models)
- **Dual LLM providers**: Groq (fast, hosted, primary) with automatic fallback to a local Ollama model if Groq is unavailable
- **Semantic search**: ChromaDB with cosine similarity over sentence-transformer embeddings
- **REST API**: `/health`, `/ingest`, `/chat`, `/reset` — usable from curl, Postman, or another service, not just the chat UI
- **API-key auth**: every mutating/expensive endpoint requires an `X-API-Key` header
- **Source citations**: every answer returns which document(s) and chunk(s) it came from, with a relevance score
- **Dockerized**: one image, two entry points (API or UI) via `docker-compose`
- **CI**: GitHub Actions runs the full pytest suite (fully mocked — no API keys needed) on every push and PR

## Prerequisites

- Python 3.11+
- (Optional) [Ollama](https://ollama.ai) installed locally if you want the offline LLM fallback
- Free API keys:
  - [Groq](https://console.groq.com/keys) — LLM generation
  - [HuggingFace](https://huggingface.co/settings/tokens) — image captioning (only needed if you'll ingest images)

## Local setup

```bash
# 1. Clone your repo (after you've pushed it — instructions below)
git clone https://github.com/ayushic353/enterprise-rag-platform.git
cd enterprise-rag-platform

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Configure environment variables
cp .env.example .env
# Open .env and fill in:
#   GROQ_API_KEY=...
#   HF_API_TOKEN=...
#   API_KEYS=<generate one, see below>

# Generate a real API key:
python -c "import secrets; print(secrets.token_urlsafe(32))"
# paste the output into API_KEYS= in .env
```

## Running it

**Option A — Streamlit chat UI**

```bash
streamlit run app.py
```
Open http://localhost:8501, upload documents in the sidebar, click "Ingest Documents", then chat.

**Option B — REST API**

```bash
uvicorn api.main:app --reload --port 8000
```
Interactive docs at http://localhost:8000/docs. Example calls:

```bash
# Health check (no auth needed)
curl http://localhost:8000/health

# Ingest a document
curl -X POST http://localhost:8000/ingest \
  -H "X-API-Key: YOUR_KEY" \
  -F "file=@./some_report.pdf"

# Ask a question
curl -X POST http://localhost:8000/chat \
  -H "X-API-Key: YOUR_KEY" \
  -H "Content-Type: application/json" \
  -d '{"question": "What were the Q3 findings?"}'
```

**Option C — Docker (runs both API and UI)**

```bash
docker compose up --build
# API:  http://localhost:8000
# UI:   http://localhost:8501
```

## Running the tests

```bash
pytest tests/ -v
```
The suite mocks the embedding model, vector store, and LLM calls, so it runs without any API keys or network access — this is exactly what CI runs on every push.

## Deploying to GitHub

See the step-by-step instructions the assistant gave you separately, or run:

```bash
git init
git add .
git commit -m "Initial commit: Enterprise RAG Platform"
git branch -M main
git remote add origin https://github.com/ayushic353/enterprise-rag-platform.git
git push -u origin main
```

## Security notes

- `.env` is git-ignored — never commit real API keys.
- `API_KEYS` in `.env.example` is intentionally blank. Generate your own with `secrets.token_urlsafe(32)`; don't reuse any example key from a README.
- `data/` and `chroma_db/` are git-ignored — they're runtime artifacts, not source.
- CORS in `api/main.py` is wide-open (`allow_origins=["*"]`) for local development — restrict it to your real frontend origin before any real deployment.

## Roadmap / possible next steps

- Swap the sliding-window chunker for a semantic/sentence-aware splitter
- Add streaming responses (SSE) from `/chat`
- Add per-user document namespaces instead of one shared collection
- Swap ChromaDB for a managed vector DB (Pinecone/Qdrant Cloud) for multi-instance deployments

## License

MIT — see [LICENSE](LICENSE).
