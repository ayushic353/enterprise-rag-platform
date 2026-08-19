"""
Thin wrapper around a persistent ChromaDB collection. Keeps Chroma's API
surface out of the rest of the codebase so it can be swapped later
(e.g. for Qdrant/Pinecone) without touching ingestion or retrieval logic.
"""
import uuid
from typing import Dict, List, Optional

import chromadb
from chromadb.config import Settings

from rag import config


class VectorStore:
    def __init__(self):
        self._client = chromadb.PersistentClient(
            path=str(config.CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: List[Dict],
        ids: Optional[List[str]] = None,
    ) -> None:
        if not texts:
            return
        ids = ids or [str(uuid.uuid4()) for _ in texts]
        self._collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

    def query(self, query_embedding: List[float], top_k: int) -> List[Dict]:
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        hits = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            hits.append({"text": doc, "metadata": meta, "distance": dist})
        return hits

    def count(self) -> int:
        return self._collection.count()

    def list_sources(self) -> List[str]:
        data = self._collection.get(include=["metadatas"])
        sources = {m.get("source", "unknown") for m in data.get("metadatas", [])}
        return sorted(sources)

    def reset(self) -> None:
        self._client.delete_collection(config.COLLECTION_NAME)
        self._collection = self._client.get_or_create_collection(
            name=config.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )


_store: Optional[VectorStore] = None


def get_store() -> VectorStore:
    global _store
    if _store is None:
        _store = VectorStore()
    return _store
