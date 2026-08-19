from typing import List, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")
    top_k: Optional[int] = Field(default=None, ge=1, le=20)


class SourceHit(BaseModel):
    source: str
    chunk_index: Optional[int]
    relevance: Optional[float]
    excerpt: str


class ChatResponse(BaseModel):
    question: str
    answer: str
    provider: str
    sources: List[SourceHit]


class IngestResponse(BaseModel):
    filename: str
    chunks_ingested: int


class HealthResponse(BaseModel):
    status: str
    documents_indexed: int
    sources: List[str]
