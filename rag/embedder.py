"""
Wraps a local sentence-transformers model so the rest of the codebase
never talks to the embedding library directly. Loaded once and cached.
"""
from functools import lru_cache
from typing import List

from sentence_transformers import SentenceTransformer

from rag import config


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    return SentenceTransformer(config.EMBEDDING_MODEL)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """Embed a batch of strings. Returns a list of float vectors."""
    if not texts:
        return []
    model = _get_model()
    vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
    return vectors.tolist()


def embed_query(text: str) -> List[float]:
    return embed_texts([text])[0]
