"""PDF chunk search — the runnable extract of `search_engine_sample.ipynb`.

Two ways to run it:

    # 1. Standalone CLI — no HTTP, no server. Drives SearchService directly.
    python search_engine_sample.py ingest pdfs/LLM.pdf
    python search_engine_sample.py search "language model"

    # 2. FastAPI service
    python search_engine_sample.py serve            # → http://127.0.0.1:9321/docs
    uvicorn search_engine_sample:app --reload --port 9321

Configuration comes from the environment (see `Settings.from_env`):

    SEARCH_DATA_DIR       where chunk CSVs are written   (default ./csv_files)
    SEARCH_PDF_ROOT       ingestion is confined here     (default cwd)
    SEARCH_CHUNK_SIZE_WORDS / SEARCH_OVERLAP_WORDS

This file does not touch `search_engine.py` or `search_ui.py`; it stands alone. 
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
import threading
import time
import traceback
import unicodedata
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import pandas as pd
import uvicorn
from fastapi import Depends, FastAPI, Query, Request, status
from fastapi.responses import JSONResponse, Response
from PyPDF2 import PdfReader
from pydantic import BaseModel, Field, field_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

log = logging.getLogger("search_engine")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #

class ConfigError(RuntimeError):
    """Raised when settings are internally inconsistent."""


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application configuration."""

    data_dir: Path            # where chunk CSVs are written
    pdf_root: Path            # ingestion is confined to this subtree
    chunk_size_words: int = 120
    overlap_words: int = 20
    max_pdf_bytes: int = 50 * 1024 * 1024
    max_pages: int = 2_000
    max_query_terms: int = 32
    default_page_size: int = 20
    max_page_size: int = 200

    def __post_init__(self) -> None:
        if self.chunk_size_words < 1:
            raise ConfigError("chunk_size_words must be >= 1")
        if not 0 <= self.overlap_words < self.chunk_size_words:
            raise ConfigError(
                f"overlap_words must satisfy 0 <= overlap < chunk_size "
                f"(got overlap={self.overlap_words}, chunk_size={self.chunk_size_words})"
            )
        if self.max_pdf_bytes < 1:
            raise ConfigError("max_pdf_bytes must be >= 1")
        if self.default_page_size > self.max_page_size:
            raise ConfigError("default_page_size cannot exceed max_page_size")

    @classmethod
    def from_env(cls, **overrides: Any) -> "Settings":
        base = Path(os.environ.get("SEARCH_DATA_DIR", Path.cwd() / "csv_files"))
        values: dict[str, Any] = {
            "data_dir": Path(overrides.pop("data_dir", base)),
            "pdf_root": Path(overrides.pop("pdf_root", os.environ.get("SEARCH_PDF_ROOT", Path.cwd()))),
            "chunk_size_words": int(os.environ.get("SEARCH_CHUNK_SIZE_WORDS", 120)),
            "overlap_words": int(os.environ.get("SEARCH_OVERLAP_WORDS", 20)),
        }
        values.update(overrides)
        return cls(**values)

    def ensure_dirs(self) -> "Settings":
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self


# --------------------------------------------------------------------------- #
# Domain errors — each carries the HTTP status the API layer maps it to
# --------------------------------------------------------------------------- #

class SearchEngineError(Exception):
    """Base class for all domain errors."""

    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    code: str = "internal_error"


class DocumentNotFound(SearchEngineError):
    status_code = status.HTTP_404_NOT_FOUND
    code = "document_not_found"


class UnsafePath(SearchEngineError):
    """Requested path escapes the configured root."""

    status_code = status.HTTP_403_FORBIDDEN
    code = "unsafe_path"


class DocumentTooLarge(SearchEngineError):
    status_code = status.HTTP_413_REQUEST_ENTITY_TOO_LARGE
    code = "document_too_large"


class UnreadableDocument(SearchEngineError):
    """File exists but is not a parseable PDF, or contains no extractable text."""

    status_code = status.HTTP_422_UNPROCESSABLE_ENTITY
    code = "unreadable_document"


class InvalidQuery(SearchEngineError):
    status_code = status.HTTP_400_BAD_REQUEST
    code = "invalid_query"


# --------------------------------------------------------------------------- #
# Safe path resolution
# --------------------------------------------------------------------------- #

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")


def resolve_within(root: Path, candidate: Path | str) -> Path:
    """Resolve `candidate` and assert it lives under `root`.

    Raises:
        UnsafePath: if the resolved path escapes `root`.
    """
    root_resolved = Path(root).resolve()
    target = Path(candidate)
    if not target.is_absolute():
        target = root_resolved / target
    target = target.resolve()
    # Compare resolved paths, never raw strings: `..` and symlinks are gone by now.
    if target != root_resolved and root_resolved not in target.parents:
        raise UnsafePath(f"path {candidate!r} resolves outside the permitted root")
    return target


def safe_stem(filename: str, *, max_len: int = 80) -> str:
    """Turn an arbitrary filename into a filesystem-safe slug (allowlist approach)."""
    stem = Path(filename).stem
    # Fold accents so "Résumé.pdf" and "Resume.pdf" don't collide unpredictably.
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    slug = _SLUG_STRIP.sub("-", stem.lower()).strip("-")
    return (slug[:max_len] or "document")


# --------------------------------------------------------------------------- #
# Text extraction and chunking
# --------------------------------------------------------------------------- #

_WHITESPACE = re.compile(r"\s+")


def normalise_text(raw: str) -> str:
    """Collapse whitespace and strip control characters from extracted PDF text."""
    cleaned = "".join(ch for ch in raw if ch == "\n" or ch == "\t" or unicodedata.category(ch)[0] != "C")
    return _WHITESPACE.sub(" ", cleaned).strip()


def window_words(words: Sequence[str], size: int, overlap: int) -> Iterator[list[str]]:
    """Yield overlapping fixed-size windows. Guaranteed to terminate: step >= 1."""
    step = size - overlap
    if step < 1:                      # defence in depth; Settings already enforces this
        raise ConfigError("overlap must be smaller than chunk size")
    if not words:
        return
    start = 0
    while start < len(words):
        yield list(words[start : start + size])
        if start + size >= len(words):
            break                     # final window reached the end; don't emit a tail duplicate
        start += step


@dataclass(frozen=True, slots=True)
class Chunk:
    """One retrievable unit of text, with provenance back to its source page."""

    document_id: str
    filename: str
    page_number: int
    chunk_number: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.document_id}:{self.page_number}:{self.chunk_number}"


def extract_pages(pdf_path: Path, *, max_pages: int) -> list[str]:
    """Extract normalised text per page. Blank pages are preserved as '' to keep numbering true."""
    try:
        reader = PdfReader(str(pdf_path))
        pages = reader.pages[:max_pages]
        return [normalise_text(page.extract_text() or "") for page in pages]
    except UnreadableDocument:
        raise
    except Exception as exc:  # PyPDF2 raises a wide variety of parse errors
        raise UnreadableDocument(f"could not parse {pdf_path.name}: {exc}") from exc


def chunk_document(
    pdf_path: Path,
    *,
    document_id: str,
    chunk_size_words: int,
    overlap_words: int,
    max_pages: int,
) -> list[Chunk]:
    """Extract and chunk a PDF. Page numbers are 1-based and survive blank pages."""
    pages = extract_pages(pdf_path, max_pages=max_pages)
    chunks: list[Chunk] = []
    for page_number, page_text in enumerate(pages, start=1):
        words = page_text.split()
        for chunk_number, window in enumerate(window_words(words, chunk_size_words, overlap_words), start=1):
            chunks.append(
                Chunk(
                    document_id=document_id,
                    filename=pdf_path.name,
                    page_number=page_number,
                    chunk_number=chunk_number,
                    text=" ".join(window),
                )
            )
    if not chunks:
        # A scanned/image-only PDF parses fine but yields nothing — that is a client-visible
        # condition (needs OCR), not a server fault.
        raise UnreadableDocument(
            f"{pdf_path.name} contains no extractable text (is it a scanned image?)"
        )
    return chunks


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #

CHUNK_COLUMNS = ["document_id", "filename", "page_number", "chunk_number", "text"]


class ChunkStore:
    """Persists chunks as one CSV per document and serves a cached in-memory corpus."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._lock = threading.RLock()
        self._cache: pd.DataFrame | None = None
        settings.data_dir.mkdir(parents=True, exist_ok=True)

    def _path_for(self, document_id: str) -> Path:
        return self._settings.data_dir / f"{document_id}.csv"

    def write(self, document_id: str, chunks: Sequence[Chunk]) -> Path:
        """Atomically persist a document's chunks and invalidate the corpus cache."""
        frame = pd.DataFrame(
            [
                {
                    "document_id": c.document_id,
                    "filename": c.filename,
                    "page_number": c.page_number,
                    "chunk_number": c.chunk_number,
                    "text": c.text,
                }
                for c in chunks
            ],
            columns=CHUNK_COLUMNS,
        )
        target = self._path_for(document_id)
        with self._lock:
            fd, tmp_name = tempfile.mkstemp(dir=self._settings.data_dir, suffix=".tmp")
            os.close(fd)
            tmp = Path(tmp_name)
            try:
                frame.to_csv(tmp, index=False)
                os.replace(tmp, target)   # atomic: readers see old or new, never partial
            finally:
                tmp.unlink(missing_ok=True)
            self._cache = None
        log.info("stored %d chunks for document_id=%s", len(chunks), document_id)
        return target

    def delete(self, document_id: str) -> None:
        with self._lock:
            path = self._path_for(document_id)
            if not path.exists():
                raise DocumentNotFound(f"no document with id {document_id!r}")
            path.unlink()
            self._cache = None

    def corpus(self) -> pd.DataFrame:
        """Return all chunks as one DataFrame. Empty (but correctly typed) when no documents."""
        with self._lock:
            if self._cache is not None:
                return self._cache
            files = sorted(self._settings.data_dir.glob("*.csv"))
            if not files:
                # pd.concat([]) raises ValueError. Return a typed empty frame instead.
                self._cache = pd.DataFrame(columns=CHUNK_COLUMNS)
                return self._cache
            frames = []
            for path in files:
                try:
                    frames.append(pd.read_csv(path, dtype={"text": "string"}))
                except (pd.errors.EmptyDataError, pd.errors.ParserError):
                    log.warning("skipping unreadable corpus file %s", path.name)
            self._cache = (
                pd.concat(frames, ignore_index=True) if frames
                else pd.DataFrame(columns=CHUNK_COLUMNS)
            )
            return self._cache

    def documents(self) -> list[dict[str, Any]]:
        corpus = self.corpus()
        if corpus.empty:
            return []
        grouped = corpus.groupby(["document_id", "filename"], as_index=False).agg(
            chunk_count=("text", "size"), page_count=("page_number", "nunique")
        )
        return grouped.to_dict("records")


# --------------------------------------------------------------------------- #
# Search
# --------------------------------------------------------------------------- #

SearchMode = Literal["any", "all"]


def compile_terms(query: str, *, max_terms: int) -> list[tuple[str, re.Pattern[str]]]:
    """Compile each whitespace-separated term into a case-insensitive word-boundary pattern.

    Returns (original_term, pattern) pairs — the raw term is kept so results can report which
    terms matched without trying to un-escape a compiled pattern.
    """
    terms = [t for t in query.split() if t.strip()]
    if not terms:
        raise InvalidQuery("query must contain at least one search term")
    if len(terms) > max_terms:
        raise InvalidQuery(f"query has {len(terms)} terms; the maximum is {max_terms}")
    # re.escape means user input becomes a literal, never a regex operator.
    return [(t, re.compile(rf"\b{re.escape(t)}\b", re.IGNORECASE)) for t in terms]


@dataclass(frozen=True, slots=True)
class SearchHit:
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_number: int
    text: str
    score: int
    matched_terms: list[str] = field(default_factory=list)


def search_corpus(
    corpus: pd.DataFrame,
    query: str,
    *,
    mode: SearchMode = "any",
    max_terms: int = 32,
) -> list[SearchHit]:
    """Rank chunks by distinct-term coverage, then by total occurrences."""
    compiled = compile_terms(query, max_terms=max_terms)
    if corpus.empty:
        return []

    hits: list[SearchHit] = []
    for row in corpus.itertuples(index=False):
        text = str(row.text)
        matched, occurrences = [], 0
        for term, pattern in compiled:
            found = pattern.findall(text)
            if found:
                matched.append(term)
                occurrences += len(found)
        if not matched:
            continue
        if mode == "all" and len(matched) != len(compiled):
            continue
        hits.append(
            SearchHit(
                chunk_id=f"{row.document_id}:{row.page_number}:{row.chunk_number}",
                document_id=str(row.document_id),
                filename=str(row.filename),
                page_number=int(row.page_number),
                chunk_number=int(row.chunk_number),
                text=text,
                score=len(matched) * 1000 + occurrences,   # coverage dominates frequency
                matched_terms=matched,
            )
        )
    # Stable, deterministic ordering — identical queries always return identical pages.
    hits.sort(key=lambda h: (-h.score, h.filename, h.page_number, h.chunk_number))
    return hits


# --------------------------------------------------------------------------- #
# Service layer — usable with no web framework in sight
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, slots=True)
class IngestResult:
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class SearchService:
    def __init__(self, settings: Settings, store: ChunkStore | None = None) -> None:
        self.settings = settings
        self.store = store or ChunkStore(settings)

    @staticmethod
    def _document_id(path: Path) -> str:
        # Deterministic: same path → same id → re-ingest overwrites instead of duplicating.
        return f"{safe_stem(path.name)}-{uuid.uuid5(uuid.NAMESPACE_URL, str(path)).hex[:8]}"

    def ingest(self, pdf_path: str, *, chunk_size_words: int | None = None,
               overlap_words: int | None = None) -> IngestResult:
        resolved = resolve_within(self.settings.pdf_root, pdf_path)
        if not resolved.is_file():
            raise DocumentNotFound(f"no such file: {pdf_path}")

        size = resolved.stat().st_size
        if size > self.settings.max_pdf_bytes:
            raise DocumentTooLarge(
                f"{resolved.name} is {size} bytes; the limit is {self.settings.max_pdf_bytes}"
            )
        if resolved.suffix.lower() != ".pdf":
            raise UnreadableDocument(f"{resolved.name} is not a .pdf file")

        size_words = chunk_size_words or self.settings.chunk_size_words
        overlap = self.settings.overlap_words if overlap_words is None else overlap_words
        if overlap >= size_words:
            raise InvalidQuery("overlap_words must be smaller than chunk_size_words")

        document_id = self._document_id(resolved)
        chunks = chunk_document(
            resolved,
            document_id=document_id,
            chunk_size_words=size_words,
            overlap_words=overlap,
            max_pages=self.settings.max_pages,
        )
        self.store.write(document_id, chunks)
        return IngestResult(
            document_id=document_id,
            filename=resolved.name,
            page_count=len({c.page_number for c in chunks}),
            chunk_count=len(chunks),
        )

    def search(self, query: str, *, mode: SearchMode = "any",
               limit: int = 20, offset: int = 0) -> tuple[list[SearchHit], int]:
        """Return one page of hits plus the total match count."""
        hits = search_corpus(
            self.store.corpus(), query, mode=mode, max_terms=self.settings.max_query_terms
        )
        return hits[offset : offset + limit], len(hits)

    def documents(self) -> list[dict[str, Any]]:
        return self.store.documents()

    def delete(self, document_id: str) -> None:
        self.store.delete(document_id)


# --------------------------------------------------------------------------- #
# API schemas
# --------------------------------------------------------------------------- #

class IngestRequest(BaseModel):
    pdf_path: str = Field(..., min_length=1, description="Path to a PDF, relative to the configured root")
    chunk_size_words: int | None = Field(None, ge=10, le=2000)
    overlap_words: int | None = Field(None, ge=0, le=500)

    @field_validator("pdf_path")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("pdf_path must not be blank")
        return v.strip()


class IngestResponse(BaseModel):
    document_id: str
    filename: str
    page_count: int
    chunk_count: int


class SearchHitModel(BaseModel):
    chunk_id: str
    document_id: str
    filename: str
    page_number: int
    chunk_number: int
    text: str
    score: int
    matched_terms: list[str]


class SearchResponse(BaseModel):
    query: str
    mode: SearchMode
    total: int
    limit: int
    offset: int
    results: list[SearchHitModel]


class DocumentSummary(BaseModel):
    document_id: str
    filename: str
    chunk_count: int
    page_count: int


class HealthResponse(BaseModel):
    status: Literal["healthy"]
    documents: int
    chunks: int


class ErrorResponse(BaseModel):
    code: str
    detail: str
    request_id: str


# --------------------------------------------------------------------------- #
# Application factory
# --------------------------------------------------------------------------- #

def build_app(settings: Settings) -> FastAPI:
    settings.ensure_dirs()
    service = SearchService(settings)

    app = FastAPI(
        title="PDF Chunk Search",
        version="2.0.0",
        summary="Chunk PDFs into retrievable passages and search them by keyword.",
    )
    app.state.service = service

    def get_service() -> SearchService:
        return app.state.service

    @app.middleware("http")
    async def add_request_id(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
        request.state.request_id = request_id
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Request-ID"] = request_id
        log.info(
            "%s %s → %s (%.1f ms) [%s]",
            request.method, request.url.path, response.status_code, elapsed_ms, request_id,
        )
        return response

    @app.exception_handler(SearchEngineError)
    async def handle_domain_error(request: Request, exc: SearchEngineError):
        request_id = getattr(request.state, "request_id", "-")
        log.warning("%s: %s [%s]", exc.code, exc, request_id)
        return JSONResponse(
            status_code=exc.status_code,
            content=ErrorResponse(code=exc.code, detail=str(exc), request_id=request_id).model_dump(),
        )

    @app.exception_handler(Exception)
    async def handle_unexpected(request: Request, exc: Exception):
        request_id = getattr(request.state, "request_id", "-")
        # Full detail to the log, generic message to the client.
        log.error("unhandled error [%s]\n%s", request_id, traceback.format_exc())
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                code="internal_error", detail="An unexpected error occurred.", request_id=request_id
            ).model_dump(),
        )

    @app.get("/health", response_model=HealthResponse, tags=["ops"])
    async def health(svc: SearchService = Depends(get_service)) -> HealthResponse:
        corpus = svc.store.corpus()
        return HealthResponse(
            status="healthy",
            documents=int(corpus["document_id"].nunique()) if not corpus.empty else 0,
            chunks=int(len(corpus)),
        )

    @app.post("/documents", response_model=IngestResponse,
              status_code=status.HTTP_201_CREATED, tags=["documents"])
    async def ingest_document(
        payload: IngestRequest, svc: SearchService = Depends(get_service)
    ) -> IngestResponse:
        result = svc.ingest(
            payload.pdf_path,
            chunk_size_words=payload.chunk_size_words,
            overlap_words=payload.overlap_words,
        )
        # asdict, not __dict__: these dataclasses use slots=True and so have no __dict__.
        return IngestResponse(**asdict(result))

    @app.get("/documents", response_model=list[DocumentSummary], tags=["documents"])
    async def list_documents(svc: SearchService = Depends(get_service)) -> list[DocumentSummary]:
        return [DocumentSummary(**d) for d in svc.documents()]

    # response_class=Response (not the default JSONResponse) because 204 must have no body —
    # FastAPI refuses to build the route otherwise.
    @app.delete("/documents/{document_id}", status_code=status.HTTP_204_NO_CONTENT,
                response_class=Response, tags=["documents"])
    async def delete_document(document_id: str, svc: SearchService = Depends(get_service)) -> Response:
        svc.delete(document_id)
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    @app.get("/search", response_model=SearchResponse, tags=["search"])
    async def search(
        q: str = Query(..., min_length=1, description="Whitespace-separated search terms"),
        mode: SearchMode = Query("any", description="'any' = OR across terms, 'all' = AND"),
        limit: int = Query(settings.default_page_size, ge=1, le=settings.max_page_size),
        offset: int = Query(0, ge=0),
        svc: SearchService = Depends(get_service),
    ) -> SearchResponse:
        hits, total = svc.search(q, mode=mode, limit=limit, offset=offset)
        return SearchResponse(
            query=q, mode=mode, total=total, limit=limit, offset=offset,
            results=[SearchHitModel(**asdict(h)) for h in hits],
        )

    return app


# The ASGI entry point for `uvicorn search_engine_sample:app`.
app = build_app(Settings.from_env())


# --------------------------------------------------------------------------- #
# Standalone CLI
# --------------------------------------------------------------------------- #

def _cli(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="chunk a PDF into the corpus")
    p_ingest.add_argument("pdf_path", help="path to a PDF, relative to SEARCH_PDF_ROOT")
    p_ingest.add_argument("--chunk-size-words", type=int, default=None)
    p_ingest.add_argument("--overlap-words", type=int, default=None)

    p_search = sub.add_parser("search", help="search the ingested corpus")
    p_search.add_argument("query", help="whitespace-separated search terms")
    p_search.add_argument("--mode", choices=("any", "all"), default="any")
    p_search.add_argument("--limit", type=int, default=10)
    p_search.add_argument("--offset", type=int, default=0)

    sub.add_parser("documents", help="list ingested documents")

    p_serve = sub.add_parser("serve", help="run the FastAPI service")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=9321)
    p_serve.add_argument("--reload", action="store_true")

    args = parser.parse_args(argv)
    settings = Settings.from_env().ensure_dirs()

    if args.command == "serve":
        if args.reload:
            # --reload needs an import string so the worker can re-import this module.
            uvicorn.run("search_engine_sample:app", host=args.host, port=args.port, reload=True)
        else:
            uvicorn.run(build_app(settings), host=args.host, port=args.port)
        return 0

    service = SearchService(settings)
    try:
        if args.command == "ingest":
            result = service.ingest(
                args.pdf_path,
                chunk_size_words=args.chunk_size_words,
                overlap_words=args.overlap_words,
            )
            print(f"{result.filename}: {result.chunk_count} chunks across "
                  f"{result.page_count} page(s)  [id={result.document_id}]")

        elif args.command == "search":
            hits, total = service.search(
                args.query, mode=args.mode, limit=args.limit, offset=args.offset
            )
            print(f"{total} hit(s) for {args.query!r} (mode={args.mode}), "
                  f"showing {len(hits)} from offset {args.offset}\n")
            for hit in hits:
                snippet = hit.text[:150] + ("…" if len(hit.text) > 150 else "")
                print(f"[{hit.score:>5}] {hit.filename} p{hit.page_number}c{hit.chunk_number}")
                print(f"        {snippet}\n")

        elif args.command == "documents":
            docs = service.documents()
            if not docs:
                print("corpus is empty — run `ingest` first")
            for d in docs:
                print(f"{d['document_id']:<40} {d['filename']:<40} "
                      f"{d['chunk_count']:>5} chunks  {d['page_count']:>4} pages")

    except SearchEngineError as exc:
        print(f"error ({exc.code}): {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
