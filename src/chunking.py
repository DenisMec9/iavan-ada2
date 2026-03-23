from __future__ import annotations

from typing import Iterable


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    step = chunk_size - overlap

    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


def build_chunk_records(
    docs: Iterable[dict],
    chunk_size: int = 900,
    overlap: int = 120,
) -> list[dict]:
    records: list[dict] = []
    chunk_id = 0

    for doc_idx, doc in enumerate(docs):
        chunks = chunk_text(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for pos, chunk in enumerate(chunks):
            records.append(
                {
                    "chunk_id": chunk_id,
                    "doc_id": doc_idx,
                    "chunk_pos": pos,
                    "url": doc["url"],
                    "title": doc.get("title", ""),
                    "text": chunk,
                }
            )
            chunk_id += 1

    return records

