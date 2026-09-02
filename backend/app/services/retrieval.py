import logging
import time
import uuid
from typing import Optional

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.models import Chunk, Document
from app.services.ingestion import embed_texts

logger = logging.getLogger(__name__)


async def retrieve_chunks(
    db: AsyncSession,
    query: str,
    top_k: int | None = None,
    document_ids: Optional[list[uuid.UUID]] = None,
) -> list[dict]:
    """
    Retrieve the top-k most similar chunks for a query using pgvector
    cosine similarity.

    Returns a list of dicts with chunk data + similarity score.
    Logs retrieval latency and result count.
    """
    top_k = top_k or settings.top_k
    t_start = time.perf_counter()

    # 1. Embed query
    query_embeddings = await embed_texts([query])
    query_vec = query_embeddings[0]

    # 2. Format vector literal for pgvector
    vec_str = "[" + ",".join(str(v) for v in query_vec) + "]"

    # 3. Build and execute cosine similarity query
    # <=> is cosine distance; similarity = 1 - distance
    if document_ids:
        doc_id_strs = [str(did) for did in document_ids]
        stmt = text(
            """
            SELECT
                c.id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.section_heading,
                c.char_start,
                c.char_end,
                c.page_number,
                d.filename,
                1 - (c.embedding <=> CAST(:vec AS vector)) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.document_id = ANY(CAST(:doc_ids AS uuid[]))
            ORDER BY c.embedding <=> CAST(:vec AS vector)
            LIMIT :top_k
            """
        )
        result = await db.execute(
            stmt,
            {
                "vec": vec_str,
                "doc_ids": doc_id_strs,
                "top_k": top_k,
            },
        )
    else:
        stmt = text(
            """
            SELECT
                c.id,
                c.document_id,
                c.chunk_index,
                c.content,
                c.section_heading,
                c.char_start,
                c.char_end,
                c.page_number,
                d.filename,
                1 - (c.embedding <=> CAST(:vec AS vector)) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            ORDER BY c.embedding <=> CAST(:vec AS vector)
            LIMIT :top_k
            """
        )
        result = await db.execute(stmt, {"vec": vec_str, "top_k": top_k})

    rows = result.mappings().all()

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Retrieval: query=%r, results=%d, latency=%.3fs",
        query[:80],
        len(rows),
        elapsed,
    )

    return [
        {
            "id": str(row["id"]),
            "document_id": str(row["document_id"]),
            "chunk_index": row["chunk_index"],
            "content": row["content"],
            "section_heading": row["section_heading"],
            "char_start": row["char_start"],
            "char_end": row["char_end"],
            "page_number": row["page_number"],
            "filename": row["filename"],
            "similarity": float(row["similarity"]),
        }
        for row in rows
    ]
