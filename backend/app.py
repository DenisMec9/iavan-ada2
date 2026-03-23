from __future__ import annotations

import json
import os
import re
import unicodedata
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import numpy as np
import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.embeddings import EmbeddingEncoder, l2_normalize
from src.indexer import load_index
from src.search import search_top_k

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
INDEX_PATH = DATA_DIR / "index" / "faiss.index"
CHUNKS_PATH = DATA_DIR / "processed" / "chunks.jsonl"
FRONTEND_DIR = BASE_DIR / "frontend"

MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
METRIC = os.getenv("SEARCH_METRIC", "ip").lower()
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "local").lower()
HF_TOKEN = os.getenv("HF_TOKEN", "").strip()

app = FastAPI(title="AV02 Similarity Search API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SOURCE_LABELS = {
    "universalmusic.com.br": "Universal Music Brasil",
    "billboard.com.br": "Billboard Brasil",
    "aqueletuim.com.br": "Aquele Tuim",
    "apple.com": "Apple Music",
    "wikipedia.org": "Wikipedia",
    "shazam.com": "Shazam",
}

SOURCE_TYPE_LABELS = {
    "enciclopedia": "Base enciclopedica",
    "catalogo-musical": "Catalogo musical",
    "release-oficial": "Fonte oficial",
    "materia-critica": "Fonte jornalistica",
    "site-web": "Fonte web",
}

MOJIBAKE_REPLACEMENTS = {
    "\u00c3\u00a1": "á",
    "\u00c3\u00a0": "à",
    "\u00c3\u00a2": "â",
    "\u00c3\u00a3": "ã",
    "\u00c3\u00a9": "é",
    "\u00c3\u00aa": "ê",
    "\u00c3\u00ad": "í",
    "\u00c3\u00b3": "ó",
    "\u00c3\u00b4": "ô",
    "\u00c3\u00b5": "õ",
    "\u00c3\u00ba": "ú",
    "\u00c3\u00a7": "ç",
    "\u00c3\u0081": "Á",
    "\u00c3\u0080": "À",
    "\u00c3\u0082": "Â",
    "\u00c3\u0083": "Ã",
    "\u00c3\u0089": "É",
    "\u00c3\u008a": "Ê",
    "\u00c3\u008d": "Í",
    "\u00c3\u0093": "Ó",
    "\u00c3\u0094": "Ô",
    "\u00c3\u0095": "Õ",
    "\u00c3\u009a": "Ú",
    "\u00c3\u0087": "Ç",
    "\u00e2\u0080\u0099": "'",
    "\u00e2\u0080\u009c": '"',
    "\u00e2\u0080\u009d": '"',
    "\u00e2\u0080\u0093": "-",
    "\u00e2\u0080\u0094": "-",
    "\u00c2\u00a0": " ",
    "\u00c2": "",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def get_encoder() -> EmbeddingEncoder:
    encoder = getattr(app.state, "encoder", None)
    if encoder is None:
        app.state.encoder = EmbeddingEncoder(MODEL_NAME)
    return app.state.encoder


def _embed_query_hf_api(query: str) -> np.ndarray:
    api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{MODEL_NAME}"
    headers: dict[str, str] = {"Content-Type": "application/json"}
    if HF_TOKEN:
        headers["Authorization"] = f"Bearer {HF_TOKEN}"

    payload = {
        "inputs": query,
        "options": {"wait_for_model": True},
    }
    response = requests.post(api_url, headers=headers, json=payload, timeout=60)
    if response.status_code >= 400:
        raise HTTPException(
            status_code=503,
            detail=f"Embedding API error ({response.status_code}).",
        )

    data = response.json()
    arr = np.array(data, dtype="float32")
    if arr.ndim == 1:
        vec = arr
    elif arr.ndim == 2:
        vec = arr.mean(axis=0)
    elif arr.ndim == 3:
        vec = arr[0].mean(axis=0)
    else:
        raise HTTPException(status_code=503, detail="Invalid embedding response format.")
    return vec.reshape(1, -1).astype("float32")


def embed_query(query: str) -> np.ndarray:
    if EMBEDDING_BACKEND == "hf_api":
        return _embed_query_hf_api(query)
    encoder = get_encoder()
    return encoder.encode([query], batch_size=1)


def _count_bad_chars(text: str) -> int:
    bad_markers = ("Ã", "Â", "â", "�", "ð")
    return sum(text.count(marker) for marker in bad_markers)


def _try_fix_mojibake(text: str) -> str:
    if _count_bad_chars(text) == 0:
        return text

    candidates = [text]
    for src_enc in ("latin1", "cp1252"):
        try:
            candidates.append(text.encode(src_enc).decode("utf-8"))
        except Exception:
            pass

    best = min(candidates, key=_count_bad_chars)
    return best


def _clean_text(text: str) -> str:
    cleaned = _try_fix_mojibake(text)
    for bad, good in MOJIBAKE_REPLACEMENTS.items():
        cleaned = cleaned.replace(bad, good)
    cleaned = unicodedata.normalize("NFKC", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    return cleaned.strip()


def _domain_from_url(url: str) -> str:
    domain = urlparse(url).netloc.lower()
    if domain.startswith("www."):
        domain = domain[4:]
    return domain


def _source_label(domain: str) -> str:
    for key, label in SOURCE_LABELS.items():
        if key in domain:
            return label
    return domain or "fonte-desconhecida"


def _source_type(domain: str, url: str) -> str:
    joined = f"{domain} {url}".lower()
    if "wikipedia.org" in joined:
        return "enciclopedia"
    if "apple.com" in joined or "shazam.com" in joined:
        return "catalogo-musical"
    if "universalmusic.com.br" in joined:
        return "release-oficial"
    if "billboard.com.br" in joined or "aqueletuim.com.br" in joined:
        return "materia-critica"
    return "site-web"


def _relevance_label(score: float) -> str:
    if score >= 0.45:
        return "alta"
    if score >= 0.32:
        return "media"
    return "moderada"


def _build_excerpt(text: str, query: str, max_chars: int = 360) -> str:
    clean = _clean_text(text)
    if len(clean) <= max_chars:
        return clean

    q = _clean_text(query).lower()
    clean_lower = clean.lower()
    match_idx = clean_lower.find(q) if q else -1

    if match_idx >= 0:
        start = max(0, match_idx - 90)
    else:
        start = 0

    if start > 0:
        left = clean.rfind(". ", max(0, start - 120), start)
        if left != -1:
            start = left + 2

    end = min(len(clean), start + max_chars)
    chunk = clean[start:end].strip()
    if not chunk:
        chunk = clean[:max_chars].strip()
        start = 0
        end = min(len(clean), max_chars)

    prefix = "... " if start > 0 else ""
    suffix = " ..." if end < len(clean) else ""
    return f"{prefix}{chunk}{suffix}"


def _sanitize_title(title: str, url: str) -> str:
    clean_title = _clean_text(title)
    if clean_title:
        return clean_title
    domain = _domain_from_url(url)
    return _source_label(domain)


def _is_low_quality_text(text: str) -> bool:
    clean = _clean_text(text)
    if len(clean) < 40:
        return True
    bad = _count_bad_chars(clean)
    return bad >= 6


def _present_result(row: dict[str, Any], query: str) -> dict[str, Any]:
    url = row.get("url", "")
    domain = _domain_from_url(url)
    score = float(row.get("score", 0.0))
    source_type = _source_type(domain, url)
    return {
        "rank": row.get("rank"),
        "score": score,
        "relevance": _relevance_label(score),
        "title": _sanitize_title(str(row.get("title", "")), url),
        "url": url,
        "source_name": _source_label(domain),
        "source_domain": domain,
        "source_type": source_type,
        "source_type_label": SOURCE_TYPE_LABELS.get(source_type, "Fonte web"),
        "excerpt": _build_excerpt(str(row.get("text", "")), query),
    }


@app.on_event("startup")
def startup_event() -> None:
    if not INDEX_PATH.exists():
        raise RuntimeError(f"Index not found: {INDEX_PATH}. Run build first.")
    if not CHUNKS_PATH.exists():
        raise RuntimeError(f"Chunks metadata not found: {CHUNKS_PATH}. Run build first.")
    if not FRONTEND_DIR.exists():
        raise RuntimeError(f"Frontend folder not found: {FRONTEND_DIR}.")

    app.state.index = load_index(str(INDEX_PATH))
    app.state.metadata = read_jsonl(CHUNKS_PATH)
    app.state.encoder = None


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "chunks_loaded": len(app.state.metadata),
        "metric": METRIC,
        "model": MODEL_NAME,
        "embedding_backend": EMBEDDING_BACKEND,
        "encoder_loaded": app.state.encoder is not None,
    }


@app.get("/api/search")
def search(
    q: str = Query(..., min_length=2, description="Query text"),
    top_k: int = Query(5, ge=1, le=20, description="Number of results"),
) -> dict[str, Any]:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    query_vec = embed_query(query)
    if METRIC == "ip":
        query_vec = l2_normalize(query_vec)

    raw_results = search_top_k(app.state.index, query_vec, app.state.metadata, top_k=top_k * 2)
    presented = [_present_result(row, query) for row in raw_results]
    results = [row for row in presented if not _is_low_quality_text(row.get("excerpt", ""))]
    if not results:
        results = presented
    results = results[:top_k]
    for i, row in enumerate(results, start=1):
        row["rank"] = i
    return {"query": query, "top_k": top_k, "count": len(results), "results": results}


@app.get("/")
def index_page() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "favicon.svg", media_type="image/svg+xml")


app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
