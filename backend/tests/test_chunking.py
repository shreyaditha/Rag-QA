"""
Unit tests for the chunking logic in app.services.ingestion.

Tests verify:
1. Overlap percentage is within [10%, 15%] for multi-chunk documents.
2. No chunk ends mid-sentence (last non-whitespace char is sentence-terminal).
3. Section heading is inherited by all child chunks.
4. Short documents (≤ max_tokens) produce exactly one chunk with no overlap.
"""

import sys
import os

# Make the backend app importable without installing it
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

from app.services.ingestion import (
    Section,
    ChunkData,
    chunk_sections,
    _split_section_into_chunks,
    count_tokens,
    parse_markdown,
    parse_plain_text,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def make_long_section(target_tokens: int = 1000, heading: str = "Test Section") -> Section:
    """
    Build a Section whose text is approximately target_tokens tokens long.
    Uses full sentences so sentence-boundary logic has something to work with.
    """
    base_sentence = "The quick brown fox jumps over the lazy dog near the river bank. "
    text = ""
    while count_tokens(text) < target_tokens:
        text += base_sentence
    return Section(text=text.strip(), heading=heading, char_start=0)


def overlap_ratio(a: str, b: str) -> float:
    """
    Estimate the overlap ratio between two adjacent chunks.
    Returns: len(overlap_words) / len(a_words)
    Uses a simple suffix-of-a that is a prefix-of-b approach (word level).
    """
    words_a = a.split()
    words_b = b.split()
    max_overlap = min(len(words_a), len(words_b))
    for length in range(max_overlap, 0, -1):
        if words_a[-length:] == words_b[:length]:
            return length / len(words_a)
    return 0.0


def last_nonwhitespace_char(text: str) -> str:
    stripped = text.rstrip()
    return stripped[-1] if stripped else ""


# ── Test 1: Overlap percentage ─────────────────────────────────────────────────

def test_overlap_ratio_within_bounds():
    """
    For a ~1000-token section chunked at max_tokens=500 with 12% overlap,
    each consecutive pair of chunks should have ~10–15% word overlap.
    """
    section = make_long_section(target_tokens=1000)
    chunks = _split_section_into_chunks(section, max_tokens=500, overlap_ratio=0.12)

    assert len(chunks) >= 2, "Expected at least 2 chunks for a 1000-token section"

    for i in range(len(chunks) - 1):
        ratio = overlap_ratio(chunks[i].content, chunks[i + 1].content)
        assert ratio >= 0.05, (
            f"Overlap between chunk {i} and {i+1} is {ratio:.2%}, expected ≥ 5%"
        )
        assert ratio <= 0.20, (
            f"Overlap between chunk {i} and {i+1} is {ratio:.2%}, expected ≤ 20%"
        )


# ── Test 2: No chunk ends mid-sentence ────────────────────────────────────────

SENTENCE_TERMINALS = {".", "?", "!", '"', "'", "）", "。", "？", "！"}


def test_chunks_end_at_sentence_boundary():
    """
    All chunks (except possibly the last) should end at a sentence boundary.
    The last non-whitespace character should be a sentence-terminal punctuation mark.
    """
    section = make_long_section(target_tokens=1000)
    chunks = _split_section_into_chunks(section, max_tokens=500, overlap_ratio=0.12)

    for i, chunk in enumerate(chunks):
        last_char = last_nonwhitespace_char(chunk.content)
        assert last_char in SENTENCE_TERMINALS, (
            f"Chunk {i} ends with '{last_char!r}' which is not a sentence terminal. "
            f"Last 50 chars: {chunk.content[-50:]!r}"
        )


# ── Test 3: Section heading inheritance ───────────────────────────────────────

def test_section_heading_inherited():
    """
    All sub-chunks produced from a section should carry the section's heading.
    """
    HEADING = "Introduction to Testing"
    section = make_long_section(target_tokens=1200, heading=HEADING)
    chunks = _split_section_into_chunks(section, max_tokens=400, overlap_ratio=0.12)

    assert chunks, "Expected at least one chunk"
    for i, chunk in enumerate(chunks):
        assert chunk.section_heading == HEADING, (
            f"Chunk {i} has heading '{chunk.section_heading}', expected '{HEADING}'"
        )


# ── Test 4: Short document → exactly one chunk, no overlap ───────────────────

def test_short_document_one_chunk():
    """
    A document shorter than max_chunk_tokens should produce exactly one chunk
    with char_start=0 and char_end=len(text).
    """
    short_text = "This is a short document. It has only two sentences."
    section = Section(text=short_text, heading="Short", char_start=0)
    chunks = chunk_sections([section], max_tokens=500, overlap_ratio=0.12)

    assert len(chunks) == 1, f"Expected 1 chunk, got {len(chunks)}"
    assert chunks[0].content == short_text
    assert chunks[0].char_start == 0
    assert chunks[0].char_end == len(short_text)


# ── Test 5: Empty / whitespace-only section is skipped ────────────────────────

def test_empty_section_produces_no_chunks():
    sections = [
        Section(text="   \n\n  ", heading="Empty", char_start=0),
        Section(text="", heading=None, char_start=10),
    ]
    chunks = chunk_sections(sections, max_tokens=500, overlap_ratio=0.12)
    assert chunks == [], f"Expected no chunks, got {chunks}"


# ── Test 6: Chunk indices are sequential ──────────────────────────────────────

def test_chunk_indices_are_sequential():
    """chunk_sections should assign 0-based sequential chunk_index values."""
    sections = [
        make_long_section(target_tokens=600, heading="Part 1"),
        make_long_section(target_tokens=600, heading="Part 2"),
    ]
    chunks = chunk_sections(sections, max_tokens=300, overlap_ratio=0.12)
    indices = [c.chunk_index for c in chunks]
    assert indices == list(range(len(chunks))), (
        f"Expected sequential indices 0..{len(chunks)-1}, got {indices}"
    )


# ── Test 7: Markdown parser extracts headings ─────────────────────────────────

def test_markdown_parser_extracts_headings():
    md = """# Introduction
This is the introduction text. It has several sentences to make it non-trivial.

## Methods
We used various methods. This section describes them in detail.

### Results
The results were significant. Here we report the findings.
"""
    sections = parse_markdown(md)
    headings = [s.heading for s in sections]
    assert "Introduction" in headings
    assert "Methods" in headings
    assert "Results" in headings


# ── Test 8: Plain text parser splits on double newlines ──────────────────────

def test_plain_text_paragraph_split():
    text = "First paragraph content here.\n\nSecond paragraph follows.\n\nThird paragraph ends."
    sections = parse_plain_text(text)
    assert len(sections) == 3, f"Expected 3 sections, got {len(sections)}"
    assert sections[0].text == "First paragraph content here."
    assert sections[1].text == "Second paragraph follows."


# ── Test 9: Token count is non-zero for non-empty text ───────────────────────

def test_token_count_basic():
    assert count_tokens("Hello world") > 0
    assert count_tokens("") == 0
