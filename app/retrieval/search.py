"""Search screenshot OCR text in memory or through the saved FAISS index."""

from __future__ import annotations

from math import sqrt
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from .embeddings import SentenceTransformerEmbeddings
from .vector_store import FaissVectorStore, VectorStoreError


class EmbeddingProvider(Protocol):
    """Small embedding interface shared by both search paths."""

    def embed_text(self, text: str) -> list[float]:
        """Embed one query."""
        ...

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Embed several documents."""
        ...


_default_embedder: SentenceTransformerEmbeddings | None = None


def _get_default_embedder() -> SentenceTransformerEmbeddings:
    """Return the process-wide lazy-loaded local model wrapper."""
    global _default_embedder
    if _default_embedder is None:
        _default_embedder = SentenceTransformerEmbeddings()
    return _default_embedder


def _cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Calculate cosine similarity, returning zero for invalid vectors."""
    if not left or len(left) != len(right):
        return 0.0
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def search_screenshots(
    records: Sequence[Mapping[str, Any]],
    query: str,
    top_k: int = 5,
    embedder: EmbeddingProvider | None = None,
) -> list[dict[str, Any]]:
    """Rank supplied screenshot records in memory by cosine similarity.

    Kept as the direct in-memory search helper from the previous MVP step.
    Persistent operations should use ``search_index``.
    """
    if top_k <= 0 or not query.strip():
        return []
    searchable = [
        record for record in records
        if isinstance(record.get("text"), str) and record["text"].strip()
    ]
    if not searchable:
        return []
    actual_embedder = embedder or _get_default_embedder()
    texts = [str(record["text"]) for record in searchable]
    query_vector = actual_embedder.embed_text(query)
    vectors = actual_embedder.embed_texts(texts)
    if len(vectors) != len(searchable):
        raise ValueError("The embedding model returned an unexpected number of vectors.")
    ranked = [
        {
            "filename": record.get("filename") or record.get("file_name") or "",
            "path": record.get("path") or record.get("absolute_path") or "",
            "timestamp": record.get("modified_at") or record.get("created_at"),
            "text": record["text"],
            "score": _cosine_similarity(query_vector, vector),
        }
        for record, vector in zip(searchable, vectors)
    ]
    ranked.sort(key=lambda result: result["score"], reverse=True)
    return ranked[:top_k]


def search_index(
    query: str,
    top_k: int = 5,
    index_directory: str | Path | None = None,
    embedder: EmbeddingProvider | None = None,
) -> list[dict[str, Any]]:
    """Embed a query and search the saved FAISS index, returning ranked metadata.

    Missing, empty, or unreadable indexes return no results. A custom embedder
    can be supplied for deterministic tests; otherwise the local model is reused.
    """
    if top_k <= 0 or not query.strip():
        return []
    index_dir = (
        Path(index_directory)
        if index_directory is not None
        else Path(__file__).resolve().parents[2] / "data" / "index"
    )
    try:
        store = FaissVectorStore.load(index_dir / "screenshots.faiss", index_dir / "metadata.json")
    except (FileNotFoundError, VectorStoreError, OSError):
        return []
    if store.size == 0:
        return []
    actual_embedder = embedder or _get_default_embedder()
    return store.search_embeddings(actual_embedder.embed_text(query), top_k=top_k)