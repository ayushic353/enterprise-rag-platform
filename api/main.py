"""
REST API for the Enterprise RAG Platform.

Run directly:
    uvicorn api.main:app --reload --port 8000

This exposes the same rag/ pipeline that app.py (Streamlit) uses, so the
system can be consumed headlessly (curl, Postman, another service) and
not just through the chat UI.
"""
import logging

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from api.auth import require_api_key
from api.schemas import ChatRequest, ChatResponse, HealthResponse, IngestResponse
from rag import config, ingest, retrieval
from rag.vector_store import get_store

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Enterprise RAG Platform API",
    description="API-first retrieval-augmented generation over your documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this to your real frontend origin(s) in production
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse, tags=["system"])
def health():
    """Unauthenticated liveness/readiness check — safe to hit from load balancers."""
    store = get_store()
    return HealthResponse(
        status="ok",
        documents_indexed=store.count(),
        sources=store.list_sources(),
    )


@app.post(
    "/ingest",
    response_model=IngestResponse,
    tags=["ingestion"],
    dependencies=[Depends(require_api_key)],
)
async def ingest_document(file: UploadFile = File(...)):
    contents = await file.read()

    size_mb = len(contents) / (1024 * 1024)
    if size_mb > config.MAX_UPLOAD_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds {config.MAX_UPLOAD_MB}MB limit.",
        )

    try:
        chunks, metadatas = ingest.process_file(file.filename, contents)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    count = retrieval.ingest_chunks(chunks, metadatas)
    return IngestResponse(filename=file.filename, chunks_ingested=count)


@app.post(
    "/chat",
    response_model=ChatResponse,
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)
def chat(request: ChatRequest):
    top_k = request.top_k or config.TOP_K
    result = retrieval.answer_question(request.question, top_k=top_k)
    return ChatResponse(**result)


@app.delete("/reset", tags=["system"], dependencies=[Depends(require_api_key)])
def reset_index():
    """Wipe the vector store. Destructive — gated behind auth."""
    get_store().reset()
    return {"status": "collection reset"}
