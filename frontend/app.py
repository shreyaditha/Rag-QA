"""
RAG Document Q&A — Streamlit Frontend
"""

import json
import os
import time
from typing import Generator

import requests
import streamlit as st

# ── Configuration ─────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="RAG Document Q&A",
    layout="wide",
)

# ── Noir Custom Styles ─────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0d0d0d;
        color: #e0e0e0;
    }
    section[data-testid="stSidebar"] {
        background-color: #141414;
        border-right: 1px solid #2a2a2a;
    }
    .stChatMessage {
        background-color: #161616 !important;
        border: 1px solid #2a2a2a !important;
        border-radius: 3px !important;
    }
    h1, h2, h3 {
        color: #d4d4d4 !important;
        letter-spacing: 0.05em;
        border-bottom: 1px solid #2a2a2a;
        padding-bottom: 6px;
        margin-bottom: 12px;
    }
    .stButton button {
        background-color: #1a1a1a !important;
        color: #e0e0e0 !important;
        border: 1px solid #4a4a4a !important;
        border-radius: 2px !important;
        transition: background-color 0.15s ease, border-color 0.15s ease;
    }
    .stButton button:hover {
        background-color: #8a1f1f !important;
        border-color: #8a1f1f !important;
        color: #ffffff !important;
    }
    div[data-testid="stFileUploader"] {
        border: 1px dashed #3a3a3a;
        padding: 8px;
        border-radius: 2px;
    }
    hr {
        border-color: #2a2a2a !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state defaults ─────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": str, "content": str, "citations": list}

if "documents" not in st.session_state:
    st.session_state.documents = []

if "selected_doc_ids" not in st.session_state:
    st.session_state.selected_doc_ids = []


# ── API helpers ────────────────────────────────────────────────────────────────

def api_list_documents() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/documents", timeout=10)
        r.raise_for_status()
        return r.json().get("documents", [])
    except Exception as exc:
        st.sidebar.error(f"Could not fetch documents: {exc}")
        return []


def api_upload_document(file) -> dict | None:
    try:
        r = requests.post(
            f"{API_URL}/documents",
            files={"file": (file.name, file.getvalue(), "application/octet-stream")},
            timeout=120,
        )
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        st.sidebar.error(f"Upload failed: {detail}")
        return None
    except Exception as exc:
        st.sidebar.error(f"Upload error: {exc}")
        return None


def api_delete_document(doc_id: str) -> bool:
    try:
        r = requests.delete(f"{API_URL}/documents/{doc_id}", timeout=10)
        r.raise_for_status()
        return True
    except Exception as exc:
        st.sidebar.error(f"Delete failed: {exc}")
        return False


def stream_chat(query: str, doc_ids: list[str]) -> Generator[tuple[str, list], None, None]:
    """
    Stream the SSE response from POST /chat.
    Yields (accumulated_text, citations) tuples as events arrive.
    """
    payload = {
        "query": query,
        "document_ids": doc_ids if doc_ids else None,
    }
    try:
        with requests.post(
            f"{API_URL}/chat",
            json=payload,
            stream=True,
            timeout=120,
            headers={"Accept": "text/event-stream"},
        ) as resp:
            resp.raise_for_status()
            accumulated = ""
            citations = []
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
                if not line.startswith("data: "):
                    continue
                data_str = line[len("data: "):]
                if data_str == "[DONE]":
                    break
                try:
                    event = json.loads(data_str)
                except json.JSONDecodeError:
                    continue

                if event.get("type") == "text":
                    accumulated += event.get("content", "")
                    yield accumulated, citations
                elif event.get("type") == "citations":
                    citations = event.get("chunks", [])
                    yield accumulated, citations
                elif event.get("type") == "error":
                    yield event.get("content", "An error occurred."), []
                    break
    except requests.HTTPError as exc:
        detail = exc.response.json().get("detail", str(exc)) if exc.response else str(exc)
        yield f"Error: {detail}", []
    except Exception as exc:
        yield f"Connection error: {exc}", []


# ── Sidebar ────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("Document Manager")

    # ── Upload ──────────────────────────────────────────────────────────────
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader(
        "Select PDF, TXT, or MD file",
        type=["pdf", "txt", "md"],
        help="Files are chunked, embedded, and stored for retrieval.",
    )
    if uploaded_file:
        if st.button("Ingest Document", use_container_width=True):
            with st.spinner(f"Ingesting '{uploaded_file.name}'..."):
                result = api_upload_document(uploaded_file)
            if result:
                st.success(
                    f"'{result['filename']}' ingested — {result['total_chunks']} chunks."
                )
                # Refresh document list
                st.session_state.documents = api_list_documents()

    st.divider()

    # ── Document list ───────────────────────────────────────────────────────
    st.subheader("Available Documents")
    if st.button("Refresh", use_container_width=True):
        st.session_state.documents = api_list_documents()

    if not st.session_state.documents:
        st.session_state.documents = api_list_documents()

    if st.session_state.documents:
        st.caption("Select documents to scope retrieval (leave unselected to search all):")
        selected = []
        for doc in st.session_state.documents:
            col1, col2 = st.columns([5, 1])
            with col1:
                checked = st.checkbox(
                    f"**{doc['filename']}**  \n"
                    f"*{doc['file_type'].upper()} · {doc['total_chunks']} chunks*",
                    key=f"chk_{doc['id']}",
                )
            with col2:
                if st.button("Delete", key=f"del_{doc['id']}", help="Delete document"):
                    with st.spinner("Deleting..."):
                        if api_delete_document(doc["id"]):
                            st.success("Document deleted.")
                            st.session_state.documents = api_list_documents()
                            st.rerun()
            if checked:
                selected.append(doc["id"])

        st.session_state.selected_doc_ids = selected
    else:
        st.info("No records on file.")

    st.divider()
    if st.button("Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main chat area ─────────────────────────────────────────────────────────────

st.title("RAG Document Q&A")

if not st.session_state.documents:
    st.info(
        "Upload a document to begin. Supported formats: PDF, TXT, Markdown."
    )

# Render existing chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("citations"):
            with st.expander(f"{len(msg['citations'])} source citation(s)", expanded=False):
                for i, cite in enumerate(msg["citations"], 1):
                    heading = cite.get("section_heading") or "(no heading)"
                    page = cite.get("page_number")
                    page_str = f" · Page {page}" if page else ""
                    sim_pct = int(cite.get("similarity", 0) * 100)
                    st.markdown(
                        f"**[{i}] {cite['filename']}** — *{heading}{page_str}*  "
                        f"*(similarity: {sim_pct}%)*"
                    )
                    st.caption(cite.get("content_preview", "")[:300])
                    if i < len(msg["citations"]):
                        st.divider()

# Chat input
if prompt := st.chat_input("Query documents..."):
    # Show user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Stream assistant response
    with st.chat_message("assistant"):
        placeholder = st.empty()
        citations_placeholder = st.empty()
        final_text = ""
        final_citations = []

        for text, citations in stream_chat(
            prompt, st.session_state.selected_doc_ids
        ):
            final_text = text
            final_citations = citations
            placeholder.markdown(text + "▌")

        placeholder.markdown(final_text)

        # Render citations
        if final_citations:
            with citations_placeholder.expander(
                f"{len(final_citations)} source citation(s)", expanded=True
            ):
                for i, cite in enumerate(final_citations, 1):
                    heading = cite.get("section_heading") or "(no heading)"
                    page = cite.get("page_number")
                    page_str = f" · Page {page}" if page else ""
                    sim_pct = int(cite.get("similarity", 0) * 100)
                    st.markdown(
                        f"**[{i}] {cite['filename']}** — *{heading}{page_str}*  "
                        f"*(similarity: {sim_pct}%)*"
                    )
                    st.caption(cite.get("content_preview", "")[:300])
                    if i < len(final_citations):
                        st.divider()

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": final_text,
            "citations": final_citations,
        }
    )
