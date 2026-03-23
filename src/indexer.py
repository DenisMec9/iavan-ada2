from __future__ import annotations

import os

import faiss
import numpy as np


def build_index(vectors: np.ndarray, metric: str = "ip") -> faiss.Index:
    if vectors.ndim != 2:
        raise ValueError("vectors must be a 2D array")
    dim = vectors.shape[1]

    metric = metric.lower()
    if metric == "ip":
        index = faiss.IndexFlatIP(dim)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        raise ValueError("metric must be 'ip' or 'l2'")

    index.add(vectors)
    return index


def save_index(index: faiss.Index, path: str) -> None:
    # Workaround for Windows unicode path issues in faiss.write_index:
    # serialize in-memory and write bytes with Python I/O.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    serialized = faiss.serialize_index(index)
    with open(path, "wb") as f:
        f.write(serialized.tobytes())


def load_index(path: str) -> faiss.Index:
    with open(path, "rb") as f:
        blob = f.read()
    arr = np.frombuffer(blob, dtype="uint8")
    return faiss.deserialize_index(arr)
