"""
Assembles the end-to-end RAG pipeline: embed the question, retrieve
matching chunks, call the LLM, and return a structured answer with
source citations. This is the single entry point both the Streamlit UI
and the FastAPI layer call, so behavior stays identical across both.
"""
from typing import List, Optional

from rag import config, embedder, llm, vision
from rag.vector_store import get_store


def ingest_chunks(chunks: List[str], metadatas: List[dict]) -> int:
    if not chunks:
        return 0
    vectors = embedder.embed_texts(chunks)
    get_store().add(texts=chunks, embeddings=vectors, metadatas=metadatas)
    return len(chunks)


def answer_question(
    question: str,
    image_bytes: Optional[bytes] = None,
    top_k: int = config.TOP_K,
) -> dict:
    """
    Full RAG round trip. If an image is attached to the question, its
    caption is appended to the query text before retrieval so the
    search also matches visually-described content.
    """
    query_text = question
    if image_bytes:
        caption = vision.caption_image(image_bytes)
        query_text = f"{question}\n[Attached image shows: {caption}]"

    query_vector = embedder.embed_query(query_text)
    hits = get_store().query(query_vector, top_k=top_k)

    context_chunks = [h["text"] for h in hits]
    result = llm.generate_answer(question, context_chunks)

    sources = [
        {
            "source": h["metadata"].get("source", "unknown"),
            "chunk_index": h["metadata"].get("chunk_index"),
            "relevance": round(1 - h["distance"], 4) if h["distance"] is not None else None,
            "excerpt": h["text"][:300],
        }
        for h in hits
    ]

    return {
        "question": question,
        "answer": result["answer"],
        "provider": result["provider"],
        "sources": sources,
    }
