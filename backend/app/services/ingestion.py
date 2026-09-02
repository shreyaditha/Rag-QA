import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx
import nltk
import tiktoken
from pypdf import PdfReader
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.config import settings
from app.models import Document, Chunk

logger = logging.getLogger(__name__)

# Ensure NLTK punkt tokenizer is available
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", quiet=True)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

TOKENIZER = tiktoken.get_encoding("cl100k_base")

ALLOWED_TYPES = {"pdf", "txt", "md"}


@dataclass
class Section:
    """A structural section of a document (heading + body text)."""
    text: str
    heading: Optional[str] = None
    page_number: Optional[int] = None
    char_start: int = 0


@dataclass
class ChunkData:
    """A single processed chunk ready for embedding."""
    content: str
    section_heading: Optional[str]
    char_start: int
    char_end: int
    page_number: Optional[int] = None
    chunk_index: int = 0


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken cl100k_base encoding."""
    return len(TOKENIZER.encode(text))


# ── Parsing ───────────────────────────────────────────────────────────────────

def parse_pdf(file_bytes: bytes) -> list[Section]:
    """Extract text from PDF, one section per page with page numbers."""
    import io
    reader = PdfReader(io.BytesIO(file_bytes))
    sections: list[Section] = []
    char_offset = 0
    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if text:
            sections.append(
                Section(
                    text=text,
                    heading=f"Page {page_num}",
                    page_number=page_num,
                    char_start=char_offset,
                )
            )
            char_offset += len(text) + 1  # +1 for separator
    return sections


def parse_markdown(text: str) -> list[Section]:
    """Split markdown on ATX headings (# Heading), preserving heading as metadata."""
    import re
    heading_pattern = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    matches = list(heading_pattern.finditer(text))

    if not matches:
        # No headings — treat entire document as one section
        return [Section(text=text.strip(), heading=None, char_start=0)]

    sections: list[Section] = []
    char_offset = 0

    # Content before first heading
    if matches[0].start() > 0:
        pre_text = text[: matches[0].start()].strip()
        if pre_text:
            sections.append(Section(text=pre_text, heading=None, char_start=0))

    for i, match in enumerate(matches):
        heading_text = match.group(2).strip()
        body_start = match.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()
        if body:
            sections.append(
                Section(
                    text=body,
                    heading=heading_text,
                    char_start=match.start(),
                )
            )

    return sections


def parse_plain_text(text: str) -> list[Section]:
    """Split plain text on double newlines (paragraph boundaries)."""
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sections: list[Section] = []
    char_offset = 0
    for para in paragraphs:
        sections.append(Section(text=para, heading=None, char_start=char_offset))
        char_offset += len(para) + 2  # +2 for \n\n
    # If there's only one paragraph, return it as-is
    if not sections:
        sections = [Section(text=text.strip(), heading=None, char_start=0)]
    return sections


def parse_file(file_bytes: bytes, filename: str) -> tuple[list[Section], str]:
    """Parse a file into structural sections. Returns (sections, file_type)."""
    ext = filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED_TYPES:
        raise ValueError(f"Unsupported file type: .{ext}. Allowed: {ALLOWED_TYPES}")

    if ext == "pdf":
        return parse_pdf(file_bytes), "pdf"
    elif ext == "md":
        text = file_bytes.decode("utf-8", errors="replace")
        return parse_markdown(text), "md"
    else:  # txt
        text = file_bytes.decode("utf-8", errors="replace")
        return parse_plain_text(text), "txt"


# ── Chunking ──────────────────────────────────────────────────────────────────

def _split_section_into_chunks(
    section: Section,
    max_tokens: int,
    overlap_ratio: float,
) -> list[ChunkData]:
    """
    Split a single section into chunks using a sentence-boundary-aware
    sliding window.

    Algorithm:
    1. Tokenize text into sentences with NLTK.
    2. Accumulate sentences until adding the next would exceed max_tokens.
    3. Emit the current window as a chunk.
    4. Slide back by overlap_tokens worth of sentences (backtracking from the end).
    5. Repeat until all sentences are consumed.
    """
    text = section.text
    sentences = nltk.sent_tokenize(text)
    if not sentences:
        return []

    max_overlap_tokens = int(max_tokens * overlap_ratio)
    chunks: list[ChunkData] = []

    window: list[str] = []       # current window of sentences
    window_tokens: int = 0
    # Track char positions within section text
    pos = 0                       # current position in original text
    window_char_start = section.char_start

    def _make_chunk(sents: list[str], char_start: int) -> ChunkData:
        content = " ".join(sents)
        char_end = char_start + len(content)
        return ChunkData(
            content=content,
            section_heading=section.heading,
            char_start=char_start,
            char_end=char_end,
            page_number=section.page_number,
        )

    i = 0
    while i < len(sentences):
        sent = sentences[i]
        sent_tokens = count_tokens(sent)

        if window_tokens + sent_tokens > max_tokens and window:
            # Emit current window
            chunks.append(_make_chunk(window, window_char_start))

            # Build overlap: backtrack from end of window
            overlap_sents: list[str] = []
            overlap_tokens = 0
            for s in reversed(window):
                t = count_tokens(s)
                if overlap_tokens + t <= max_overlap_tokens:
                    overlap_sents.insert(0, s)
                    overlap_tokens += t
                else:
                    break

            # Compute new char_start: advance past non-overlap prefix
            non_overlap_sents = window[: len(window) - len(overlap_sents)]
            advance = len(" ".join(non_overlap_sents))
            if non_overlap_sents:
                advance += 1  # space separator
            window_char_start += advance

            window = overlap_sents
            window_tokens = overlap_tokens
            # Do NOT advance i — re-process current sentence
        else:
            window.append(sent)
            window_tokens += sent_tokens
            i += 1

    # Emit remaining sentences
    if window:
        chunks.append(_make_chunk(window, window_char_start))

    return chunks


def chunk_sections(
    sections: list[Section],
    max_tokens: int | None = None,
    overlap_ratio: float | None = None,
) -> list[ChunkData]:
    """
    Chunk all sections. Sections within the token limit become one chunk;
    larger sections are split with overlap.
    """
    max_tokens = max_tokens or settings.max_chunk_tokens
    overlap_ratio = overlap_ratio or settings.chunk_overlap_ratio

    all_chunks: list[ChunkData] = []

    for section in sections:
        if not section.text.strip():
            continue

        token_count = count_tokens(section.text)
        if token_count <= max_tokens:
            # Section fits in one chunk — no need to split
            all_chunks.append(
                ChunkData(
                    content=section.text,
                    section_heading=section.heading,
                    char_start=section.char_start,
                    char_end=section.char_start + len(section.text),
                    page_number=section.page_number,
                )
            )
        else:
            sub_chunks = _split_section_into_chunks(section, max_tokens, overlap_ratio)
            all_chunks.extend(sub_chunks)

    # Assign sequential indices
    for idx, chunk in enumerate(all_chunks):
        chunk.chunk_index = idx

    return all_chunks


# ── Embedding ─────────────────────────────────────────────────────────────────

async def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of texts using Ollama nomic-embed-text (768 dims) via local HTTP."""
    all_embeddings: list[list[float]] = []

    async with httpx.AsyncClient(timeout=60.0) as client:
        for text in texts:
            response = await client.post(
                "http://host.docker.internal:11434/api/embeddings",
                json={"model": "nomic-embed-text", "prompt": text},
            )
            response.raise_for_status()
            all_embeddings.append(response.json()["embedding"])

    return all_embeddings


# ── Ingestion pipeline ────────────────────────────────────────────────────────

async def ingest_document(
    db: AsyncSession,
    file_bytes: bytes,
    filename: str,
) -> Document:
    """Full ingestion pipeline: parse → chunk → embed → store."""
    t_start = time.perf_counter()
    logger.info("Ingesting document: %s (%d bytes)", filename, len(file_bytes))

    # 1. Parse
    sections, file_type = parse_file(file_bytes, filename)
    if not sections:
        raise ValueError(f"Document '{filename}' produced no extractable text.")
    logger.info("Parsed %d sections from '%s'", len(sections), filename)

    # 2. Chunk
    chunks = chunk_sections(sections)
    if not chunks:
        raise ValueError(f"Document '{filename}' produced no chunks after splitting.")
    logger.info("Created %d chunks from '%s'", len(chunks), filename)

    # 3. Embed
    texts = [c.content for c in chunks]
    embeddings = await embed_texts(texts)
    logger.info("Embedded %d chunks", len(embeddings))

    # 4. Persist
    doc = Document(
        id=uuid.uuid4(),
        filename=filename,
        file_type=file_type,
        total_chunks=len(chunks),
    )
    db.add(doc)
    await db.flush()  # get doc.id without committing

    chunk_orm_objects = [
        Chunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=c.chunk_index,
            content=c.content,
            section_heading=c.section_heading,
            char_start=c.char_start,
            char_end=c.char_end,
            page_number=c.page_number,
            embedding=embedding,
        )
        for c, embedding in zip(chunks, embeddings)
    ]
    db.add_all(chunk_orm_objects)
    await db.flush()

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Ingestion complete: doc_id=%s, chunks=%d, elapsed=%.2fs",
        doc.id,
        len(chunks),
        elapsed,
    )
    return doc
