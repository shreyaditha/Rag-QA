import json
import logging
from typing import AsyncGenerator

import httpx

from app.config import settings

logger = logging.getLogger(__name__)

OLLAMA_URL = "http://host.docker.internal:11434/api/generate"
OLLAMA_MODEL = "llama3.1:8b"

# Full prompt is assembled inline so the system instructions are baked into the
# single "prompt" field that Ollama's /api/generate endpoint expects.
SYSTEM_INSTRUCTIONS = """You are a precise document Q&A assistant.

Rules:
1. Answer ONLY using the provided context chunks. Do not use prior knowledge.
2. For every factual claim you make, cite the chunk(s) that support it using the
   format [Chunk N] where N is the chunk number shown in the context.
3. If the context does not contain enough information to answer the question,
   say: "I could not find relevant information in the provided documents."
4. Be concise and accurate. Do not speculate or infer beyond what the context states."""


def _format_context(chunks: list[dict]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    parts = []
    for i, chunk in enumerate(chunks, start=1):
        heading = chunk.get("section_heading") or "(no heading)"
        source = chunk.get("filename", "unknown")
        page = chunk.get("page_number")
        page_str = f", page {page}" if page else ""
        parts.append(
            f"[Chunk {i}] Source: {source}{page_str} | Section: {heading}\n"
            f"{chunk['content']}"
        )
    return "\n\n".join(parts)


async def generate_answer(
    query: str,
    chunks: list[dict],
    threshold: float | None = None,
) -> AsyncGenerator[str, None]:
    """
    Generate a streaming SSE response backed by Ollama (llama3.1:8b).

    Yields SSE-formatted strings:
      data: {"type": "text", "content": "..."}
      data: {"type": "citations", "chunks": [...]}
      data: [DONE]
    """
    threshold = threshold if threshold is not None else settings.similarity_threshold

    # Guard: if no chunk clears the similarity threshold, don't call the LLM
    if not chunks or chunks[0]["similarity"] < threshold:
        no_context_msg = (
            "I could not find relevant information in the provided documents "
            "to answer your question."
        )
        yield f"data: {json.dumps({'type': 'text', 'content': no_context_msg})}\n\n"
        yield f"data: {json.dumps({'type': 'citations', 'chunks': []})}\n\n"
        yield "data: [DONE]\n\n"
        return

    context = _format_context(chunks)
    full_prompt = (
        f"{SYSTEM_INSTRUCTIONS}\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}"
    )

    citations = [
        {
            "chunk_id": c["id"],
            "document_id": c["document_id"],
            "filename": c["filename"],
            "section_heading": c.get("section_heading"),
            "chunk_index": c["chunk_index"],
            "content_preview": c["content"][:200],
            "similarity": c["similarity"],
        }
        for c in chunks
    ]

    try:
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                OLLAMA_URL,
                json={"model": OLLAMA_MODEL, "prompt": full_prompt, "stream": True},
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk_data = json.loads(line)
                    if "response" in chunk_data:
                        yield f"data: {json.dumps({'type': 'text', 'content': chunk_data['response']})}\n\n"
                    if chunk_data.get("done"):
                        break

        yield f"data: {json.dumps({'type': 'citations', 'chunks': citations})}\n\n"
        yield "data: [DONE]\n\n"

    except httpx.HTTPStatusError as exc:
        logger.error("Ollama HTTP error: %s", exc)
        error_msg = f"Ollama returned an error ({exc.response.status_code}). Is Ollama running with llama3.1:8b pulled?"
        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        yield "data: [DONE]\n\n"
    except httpx.ConnectError:
        logger.error("Cannot reach Ollama at %s", OLLAMA_URL)
        error_msg = "Cannot connect to Ollama. Make sure Ollama is running on your machine."
        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        yield "data: [DONE]\n\n"
    except Exception as exc:
        logger.exception("Unexpected generation error: %s", exc)
        error_msg = "An unexpected error occurred while generating the answer."
        yield f"data: {json.dumps({'type': 'error', 'content': error_msg})}\n\n"
        yield "data: [DONE]\n\n"
