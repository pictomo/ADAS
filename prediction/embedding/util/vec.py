import atexit
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[3]))

from llm_provider import client

_CACHE_PATH = Path(__file__).resolve().parents[1] / "cache" / "embedding.pkl"
_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)


def _load_embedding_cache() -> dict[str, np.ndarray]:
    """埋め込みキャッシュをファイルから1回だけ読み込む。"""
    if not _CACHE_PATH.exists():
        return {}

    with _CACHE_PATH.open("rb") as cache_file:
        raw_cache = pickle.load(cache_file)

    cache: dict[str, np.ndarray] = {}
    for text, embedding in raw_cache.items():
        cache[str(text)] = np.asarray(embedding, dtype=np.float32)

    return cache


def _save_embedding_cache() -> None:
    """メモリ上の埋め込みキャッシュをファイルへ1回だけ書き戻す。"""
    tmp_path = _CACHE_PATH.with_suffix(".tmp")
    with tmp_path.open("wb") as cache_file:
        pickle.dump(_EMBEDDING_CACHE, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
    tmp_path.replace(_CACHE_PATH)


_EMBEDDING_CACHE: dict[str, np.ndarray] = _load_embedding_cache()
atexit.register(_save_embedding_cache)


def str2vec_without_cache(s: str) -> np.ndarray:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=s,
    )
    return np.array(response.data[0].embedding, dtype=np.float32)


def str2vec(s: str) -> np.ndarray:
    """永続キャッシュを挟んで文字列を埋め込み化する。"""
    if s not in _EMBEDDING_CACHE:
        _EMBEDDING_CACHE[s] = str2vec_without_cache(s)

    return _EMBEDDING_CACHE[s]


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
