from pathlib import Path
import sys

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))

from llm_provider import client


def str2vec(s: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=s,
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def vec2norm(vec: np.ndarray) -> float:
    return float(np.linalg.norm(vec))


def cos_sim(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.shape != vec2.shape:
        raise ValueError("vec1 and vec2 must have the same length")

    norm1 = vec2norm(vec1)
    norm2 = vec2norm(vec2)
    if norm1 == 0.0 or norm2 == 0.0:
        return 0.0

    return float(np.dot(vec1, vec2) / (norm1 * norm2))


def dist(vec1: np.ndarray, vec2: np.ndarray) -> float:
    if vec1.shape != vec2.shape:
        raise ValueError("vec1 and vec2 must have the same length")

    return vec2norm(vec1 - vec2)
