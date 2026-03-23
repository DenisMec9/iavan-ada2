from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from chunking import build_chunk_records
from embeddings import EmbeddingEncoder, l2_normalize
from indexer import build_index, load_index, save_index
from scraping import scrape_url
from search import search_top_k

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INDEX_DIR = DATA_DIR / "index"

RAW_DOCS_PATH = RAW_DIR / "scraped_docs.jsonl"
CHUNKS_PATH = PROCESSED_DIR / "chunks.jsonl"
EMBEDDINGS_PATH = PROCESSED_DIR / "embeddings.npy"
INDEX_PATH = INDEX_DIR / "faiss.index"


def ensure_dirs() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)


def read_urls(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"URLs file not found: {path}")
    urls: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_pipeline(args: argparse.Namespace) -> None:
    ensure_dirs()
    urls = read_urls(Path(args.urls_file))
    print(f"[build] URLs loaded: {len(urls)}")

    docs: list[dict] = []
    for url in tqdm(urls, desc="Scraping pages"):
        try:
            doc = scrape_url(url)
            if len(doc.text) < args.min_chars:
                print(f"[skip] Very short text from {url} ({len(doc.text)} chars)")
                continue
            docs.append({"url": doc.url, "title": doc.title, "text": doc.text})
        except Exception as exc:
            print(f"[error] Failed scraping {url}: {exc}")

    if not docs:
        raise RuntimeError("No valid documents after scraping.")

    write_jsonl(RAW_DOCS_PATH, docs)
    print(f"[build] Saved raw docs at: {RAW_DOCS_PATH}")

    chunks = build_chunk_records(docs, chunk_size=args.chunk_size, overlap=args.overlap)
    if not chunks:
        raise RuntimeError("No chunks generated.")

    write_jsonl(CHUNKS_PATH, chunks)
    print(f"[build] Chunks generated: {len(chunks)}")
    print(f"[build] Saved chunks at: {CHUNKS_PATH}")

    encoder = EmbeddingEncoder(args.model)
    chunk_texts = [c["text"] for c in chunks]
    vectors = encoder.encode(chunk_texts, batch_size=args.batch_size)

    if args.metric == "ip":
        vectors = l2_normalize(vectors)

    np.save(EMBEDDINGS_PATH, vectors)
    print(f"[build] Saved embeddings at: {EMBEDDINGS_PATH}")

    index = build_index(vectors, metric=args.metric)
    save_index(index, str(INDEX_PATH))
    print(f"[build] Saved index at: {INDEX_PATH}")
    print("[build] Done.")


def load_search_resources(args: argparse.Namespace):
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"Index not found: {INDEX_PATH}. Run --build-index first.")
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f"Chunks metadata not found: {CHUNKS_PATH}.")

    index = load_index(str(INDEX_PATH))
    metadata = read_jsonl(CHUNKS_PATH)
    encoder = EmbeddingEncoder(args.model)
    return index, metadata, encoder


def execute_search(
    query: str,
    args: argparse.Namespace,
    index,
    metadata: list[dict],
    encoder: EmbeddingEncoder,
) -> None:
    query_vec = encoder.encode([query], batch_size=1)
    if args.metric == "ip":
        query_vec = l2_normalize(query_vec)

    results = search_top_k(index, query_vec, metadata, top_k=args.top_k)
    if not results:
        print("No results found.")
        return

    print(f"\nQuery: {query}\n")
    for row in results:
        preview = row["text"][:250].replace("\n", " ")
        print(f"[{row['rank']}] score={row['score']:.4f} chunk_id={row['chunk_id']}")
        print(f"Title: {row['title']}")
        print(f"URL: {row['url']}")
        print(f"Text: {preview}...")
        print("-" * 80)


def run_interactive(args: argparse.Namespace) -> None:
    index, metadata, encoder = load_search_resources(args)
    print("Interactive mode. Type 'exit' to stop.")
    while True:
        query = input("\nQuery> ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit", "sair"}:
            break
        execute_search(query, args, index, metadata, encoder)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Similarity Search with LLM embeddings")
    parser.add_argument("--build-index", action="store_true", help="Build the vector index")
    parser.add_argument("--query", type=str, default="", help="Query text for retrieval")
    parser.add_argument("--interactive", action="store_true", help="Interactive search mode")
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="?",
        const=5,
        default=5,
        help="Number of chunks to return (if omitted after flag, defaults to 5)",
    )
    parser.add_argument("--urls-file", type=str, default=str(DATA_DIR / "urls_mpc.txt"))
    parser.add_argument("--chunk-size", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--min-chars", type=int, default=250)
    parser.add_argument("--metric", choices=["ip", "l2"], default="ip")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be greater than 0")

    if not args.build_index and not args.query and not args.interactive:
        raise SystemExit("Use at least one: --build-index, --query, or --interactive")

    if args.build_index:
        build_pipeline(args)

    if args.query:
        index, metadata, encoder = load_search_resources(args)
        execute_search(args.query, args, index, metadata, encoder)

    if args.interactive:
        run_interactive(args)


if __name__ == "__main__":
    main()
