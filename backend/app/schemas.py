import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel


# ── Document schemas ──────────────────────────────────────────────────────────

class DocumentResponse(BaseModel):
    id: uuid.UUID
    filename: str
    file_type: str
    total_chunks: int
    created_at: datetime

    model_config = {"from_attributes": True}


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total: int


# ── Chat schemas ──────────────────────────────────────────────────────────────

class ChatRequest(BaseModel):
    query: str
    document_ids: Optional[list[uuid.UUID]] = None  # None = search all


class ChunkCitation(BaseModel):
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    section_heading: Optional[str]
    chunk_index: int
    content_preview: str  # first 200 chars
    similarity: float


# ── Health ────────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    db: str
