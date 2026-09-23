"""Streamlit front end for `search_engine_sample.py`.

This talks to the FastAPI service over HTTP, so the service must already be running:

    # terminal 1 — the API
    cd Week1/Day_1
    SEARCH_DATA_DIR=csv_files SEARCH_PDF_ROOT=pdfs \
        uv run --project .. python search_engine_sample.py serve

    # terminal 2 — this UI
    cd Week1/Day_1
    uv run --project .. streamlit run search_ui_sample.py

Point it at a different service with:

    SEARCH_API_URL=http://127.0.0.1:9999 uv run --project .. streamlit run search_ui_sample.py

Kept separate from `search_ui.py`, which belongs to `search_engine.py` and is untouched.
"""

from __future__ import annotations

import html
import os
import re

import requests
import streamlit as st

st.set_page_config(
    page_title="PDF Chunk Search",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_API_URL = os.environ.get("SEARCH_API_URL", "http://127.0.0.1:9321")
TIMEOUT = 30


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #

class ApiError(RuntimeError):
    """A non-2xx response, carrying the service's structured error body."""

    def __init__(self, status_code: int, code: str, detail: str, request_id: str = "-") -> None:
        super().__init__(detail)
        self.status_code = status_code
        self.code = code
        self.detail = detail
        self.request_id = request_id


def _request(method: str, base_url: str, path: str, **kwargs):
    try:
        response = requests.request(method, f"{base_url.rstrip('/')}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(0, "unreachable", f"could not reach the API at {base_url}: {exc}") from exc

    if response.status_code >= 400:
        # The service returns {code, detail, request_id}; FastAPI's own validation
        # errors return {detail: [...]}. Handle both.
        try:
            body = response.json()
        except ValueError:
            body = {}
        raise ApiError(
            response.status_code,
            body.get("code", "http_error"),
            str(body.get("detail", response.text)),
            body.get("request_id", "-"),
        )
    if response.status_code == 204:
        return None
    return response.json()


def get_health(base_url: str) -> dict:
    return _request("GET", base_url, "/health")


def list_documents(base_url: str) -> list[dict]:
    return _request("GET", base_url, "/documents")


def ingest(base_url: str, pdf_path: str) -> dict:
    return _request("POST", base_url, "/documents", json={"pdf_path": pdf_path})


def delete_document(base_url: str, document_id: str) -> None:
    _request("DELETE", base_url, f"/documents/{document_id}")


def search(base_url: str, query: str, mode: str, limit: int, offset: int) -> dict:
    return _request(
        "GET", base_url, "/search",
        params={"q": query, "mode": mode, "limit": limit, "offset": offset},
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #

def highlight(text: str, terms: list[str]) -> str:
    """Escape the chunk text, then mark the matched terms.

    Escaping happens first, so PDF text containing '<' or '&' can never inject markup.
    """
    escaped = html.escape(text)
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(rf"\b{re.escape(html.escape(term))}\b", re.IGNORECASE)
        escaped = pattern.sub(lambda m: f"<mark>{m.group(0)}</mark>", escaped)
    return escaped


def render_hit(hit: dict) -> None:
    header = (
        f"📄 {hit['filename']} — page {hit['page_number']}, "
        f"chunk {hit['chunk_number']}  ·  score {hit['score']}"
    )
    with st.expander(header, expanded=True):
        st.markdown(
            f"<div style='border:1px solid rgba(128,128,128,.35);border-radius:6px;"
            f"padding:12px;line-height:1.55'>{highlight(hit['text'], hit['matched_terms'])}</div>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"matched: {', '.join(hit['matched_terms'])}  |  "
            f"document_id: {hit['document_id']}  |  chunk_id: {hit['chunk_id']}"
        )


def show_api_error(exc: ApiError) -> None:
    if exc.code == "unreachable":
        st.error(f"{exc.detail}\n\nStart it with: `python search_engine_sample.py serve`")
    else:
        st.error(f"{exc.code} (HTTP {exc.status_code}): {exc.detail}")
        if exc.request_id != "-":
            st.caption(f"request id: {exc.request_id} — grep the service log for this")


# --------------------------------------------------------------------------- #
# Sidebar: connection, corpus, ingestion
# --------------------------------------------------------------------------- #

def sidebar() -> str:
    st.sidebar.header("Service")
    base_url = st.sidebar.text_input("API URL", value=DEFAULT_API_URL)

    try:
        health = get_health(base_url)
        st.sidebar.success(
            f"connected — {health['documents']} document(s), {health['chunks']} chunk(s)"
        )
    except ApiError as exc:
        st.sidebar.error("service unreachable" if exc.code == "unreachable" else exc.detail)
        st.sidebar.caption(f"expected at {base_url}")
        return base_url

    st.sidebar.divider()
    st.sidebar.header("Ingest a PDF")
    st.sidebar.caption("Path is relative to the service's SEARCH_PDF_ROOT.")
    pdf_path = st.sidebar.text_input("PDF path", placeholder="LLM.pdf")
    if st.sidebar.button("Ingest", type="primary", use_container_width=True):
        if not pdf_path.strip():
            st.sidebar.warning("Enter a path first.")
        else:
            try:
                with st.spinner(f"chunking {pdf_path}…"):
                    result = ingest(base_url, pdf_path.strip())
                st.sidebar.success(
                    f"{result['filename']}: {result['chunk_count']} chunks "
                    f"across {result['page_count']} page(s)"
                )
                st.rerun()
            except ApiError as exc:
                st.sidebar.error(f"{exc.code}: {exc.detail}")

    st.sidebar.divider()
    st.sidebar.header("Corpus")
    try:
        documents = list_documents(base_url)
    except ApiError as exc:
        st.sidebar.error(exc.detail)
        return base_url

    if not documents:
        st.sidebar.info("No documents yet — ingest one above.")
    for doc in documents:
        left, right = st.sidebar.columns([4, 1])
        left.write(f"**{doc['filename']}**")
        left.caption(f"{doc['chunk_count']} chunks · {doc['page_count']} pages")
        if right.button("🗑", key=f"del-{doc['document_id']}", help="remove from the corpus"):
            try:
                delete_document(base_url, doc["document_id"])
                st.rerun()
            except ApiError as exc:
                st.sidebar.error(exc.detail)

    return base_url


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #

def main() -> None:
    base_url = sidebar()

    st.title("📄 PDF Chunk Search")
    st.markdown(
        "Search the chunks ingested by `search_engine_sample.py`. "
        "Results are ranked by how many distinct query terms a chunk contains, "
        "then by how often they occur."
    )

    col_query, col_mode, col_limit = st.columns([4, 1, 1])
    query = col_query.text_input("Search terms", placeholder="e.g. language model")
    mode = col_mode.selectbox("Match", options=["any", "all"],
                              help="'any' = OR across terms, 'all' = AND")
    limit = col_limit.number_input("Per page", min_value=1, max_value=200, value=10)

    # Reset paging whenever the query itself changes.
    if st.session_state.get("_last_query") != (query, mode, limit):
        st.session_state["_last_query"] = (query, mode, limit)
        st.session_state["offset"] = 0

    if not query.strip():
        st.info("Enter a search term above. Ingest a PDF from the sidebar first if the corpus is empty.")
        return

    offset = st.session_state.get("offset", 0)
    try:
        with st.spinner(f"searching for {query!r}…"):
            payload = search(base_url, query.strip(), mode, int(limit), offset)
    except ApiError as exc:
        show_api_error(exc)
        return

    total = payload["total"]
    results = payload["results"]
    if total == 0:
        st.warning(f"No chunks matched {query!r} in mode '{mode}'.")
        return

    shown_from = offset + 1
    shown_to = offset + len(results)
    st.success(f"{total} matching chunk(s) — showing {shown_from}–{shown_to}")

    for hit in results:
        render_hit(hit)

    prev_col, _, next_col = st.columns([1, 6, 1])
    if prev_col.button("← Previous", disabled=offset == 0, use_container_width=True):
        st.session_state["offset"] = max(0, offset - int(limit))
        st.rerun()
    if next_col.button("Next →", disabled=shown_to >= total, use_container_width=True):
        st.session_state["offset"] = offset + int(limit)
        st.rerun()


if __name__ == "__main__":
    main()
