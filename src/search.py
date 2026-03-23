from __future__ import annotations

import faiss
import numpy as np


def search_top_k(
    index: faiss.Index,
    query_vector: np.ndarray,
    metadata: list[dict],
    top_k: int = 5,
) -> list[dict]:
    if query_vector.ndim != 2 or query_vector.shape[0] != 1:
        raise ValueError("query_vector must have shape (1, dim)")

    effective_k = max(1, min(top_k, len(metadata)))
    scores, ids = index.search(query_vector, effective_k)
    scored_results: list[dict] = []

    for idx, score in zip(ids[0], scores[0]):
        if idx < 0 or idx >= len(metadata):
            continue
        row = metadata[idx]
        scored_results.append(
            {
                "rank": len(scored_results) + 1,
                "score": float(score),
                "chunk_id": row["chunk_id"],
                "url": row["url"],
                "title": row.get("title", ""),
                "text": row["text"],
            }
        )

    return scored_results
