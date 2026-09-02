import logging

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.schemas import ChatRequest
from app.services.retrieval import retrieve_chunks
from app.services.generation import generate_answer

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("")
async def chat(
    request: ChatRequest,
    db: AsyncSession = Depends(get_db),
):
    """
    RAG chat endpoint. Returns a Server-Sent Events stream.

    SSE event format:
      data: {"type": "text", "content": "<delta>"}
      data: {"type": "citations", "chunks": [{...}, ...]}
      data: [DONE]
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    logger.info("Chat query: %r (doc_ids=%s)", request.query[:80], request.document_ids)

    try:
        chunks = await retrieve_chunks(
            db,
            query=request.query,
            document_ids=request.document_ids,
        )
    except Exception as exc:
        logger.exception("Retrieval failed: %s", exc)
        raise HTTPException(status_code=500, detail="Retrieval failed. Check server logs.")

    async def event_generator():
        async for event in generate_answer(request.query, chunks):
            yield event

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
