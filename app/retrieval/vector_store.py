"""Persistent local FAISS storage for screenshot embeddings and metadata."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence


class VectorStoreError(RuntimeError):
    """Raised when a FAISS index or its metadata cannot be used."""


def _faiss_module() -> Any:
    """Import FAISS on demand and explain how to install it if unavailable."""
    try:
        import faiss
    except ImportError as error:
        raise VectorStoreError("FAISS is missing. Install it with: python -m pip install -r requirements.txt") from error
    return faiss


def _numpy_module() -> Any:
    """Import NumPy, which FAISS uses for vector arrays."""
    try:
        import numpy as np
    except ImportError as error:
        raise VectorStoreError("NumPy is missing. Install FAISS with: python -m pip install -r requirements.txt") from error
    return np


def _stable_path(metadata: Mapping[str, Any]) -> str:
    """Return a normalized screenshot path used to identify an indexed record."""
    path = str(metadata.get("path") or metadata.get("filename") or metadata.get("file_name") or "")
    return os.path.normcase(os.path.normpath(path))


class FaissVectorStore:
    """A small IndexFlatIP store with JSON metadata kept in matching order.

    Vectors are normalized before insertion and query. Inner product between
    unit vectors is cosine similarity. FAISS stores only vectors; metadata is
    persisted separately as a JSON list with the same order as the FAISS ids.
    """

    def __init__(self) -> None:
        """Create an empty store; call ``create_index`` before adding vectors."""
        self.index: Any | None = None
        self.metadata: list[dict[str, Any]] = []

    @property
    def size(self) -> int:
        """Return the number of vectors stored in the current index."""
        return int(self.index.ntotal) if self.index is not None else 0

    @property
    def dimension(self) -> int | None:
        """Return the vector dimension, if an index has been created."""
        return int(self.index.d) if self.index is not None else None

    def create_index(self, dimension: int) -> None:
        """Create an empty exact-search FAISS index for vectors of this size."""
        if dimension <= 0:
            raise ValueError("Embedding dimension must be positive.")
        self.index = _faiss_module().IndexFlatIP(dimension)
        self.metadata = []

    def add_embeddings(
        self,
        embeddings: Sequence[Sequence[float]],
        metadata: Sequence[Mapping[str, Any]],
    ) -> None:
        """Normalize and upsert vectors by screenshot path, preserving metadata order."""
        if len(embeddings) != len(metadata):
            raise ValueError("Every embedding must have exactly one metadata record.")
        if not embeddings:
            return

        np = _numpy_module()
        faiss = _faiss_module()
        vectors = np.asarray(embeddings, dtype="float32")
        if vectors.ndim != 2 or vectors.shape[1] == 0:
            raise ValueError("Embeddings must be a two-dimensional array with non-empty vectors.")
        if self.index is None:
            self.create_index(int(vectors.shape[1]))
        if int(vectors.shape[1]) != self.dimension:
            raise ValueError(f"Embedding dimension {vectors.shape[1]} does not match index dimension {self.dimension}.")
        if self.size != len(self.metadata):
            raise VectorStoreError("FAISS vector count does not match metadata count.")
        vectors = np.ascontiguousarray(vectors)
        faiss.normalize_L2(vectors)

        existing_vectors: list[Any] = []
        if self.index is not None:
            existing_vectors = [self.index.reconstruct(row) for row in range(self.size)]
        combined_metadata = [dict(item) for item in self.metadata]
        positions = {_stable_path(item): position for position, item in enumerate(combined_metadata)}

        for vector, item in zip(vectors, metadata):
            copied_metadata = dict(item)
            stable_path = _stable_path(copied_metadata)
            if not stable_path:
                raise ValueError("Metadata must include a stable screenshot path or filename.")
            if stable_path in positions:
                position = positions[stable_path]
                combined_metadata[position] = copied_metadata
                existing_vectors[position] = vector.copy()
            else:
                positions[stable_path] = len(combined_metadata)
                combined_metadata.append(copied_metadata)
                existing_vectors.append(vector.copy())

        replacement = faiss.IndexFlatIP(int(self.dimension))
        if existing_vectors:
            matrix = np.ascontiguousarray(np.asarray(existing_vectors, dtype="float32"))
            faiss.normalize_L2(matrix)
            replacement.add(matrix)
        self.index = replacement
        self.metadata = combined_metadata

    def retain_paths(self, paths: set[str]) -> None:
        """Remove records whose normalized paths are not in ``paths``."""
        if self.index is None:
            return
        keep_positions = [
            position for position, record in enumerate(self.metadata)
            if _stable_path(record) in paths
        ]
        if len(keep_positions) == self.size:
            return
        np = _numpy_module()
        faiss = _faiss_module()
        vectors = [self.index.reconstruct(position) for position in keep_positions]
        self.metadata = [self.metadata[position] for position in keep_positions]
        self.index = faiss.IndexFlatIP(int(self.dimension))
        if vectors:
            matrix = np.ascontiguousarray(np.asarray(vectors, dtype="float32"))
            faiss.normalize_L2(matrix)
            self.index.add(matrix)

    def search_embeddings(
        self,
        embedding: Sequence[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search with a normalized query and return metadata with cosine scores."""
        if top_k <= 0 or self.size == 0 or self.index is None:
            return []
        np = _numpy_module()
        faiss = _faiss_module()
        query = np.asarray([embedding], dtype="float32")
        if query.ndim != 2 or query.shape[1] != self.dimension:
            raise ValueError(f"Query embedding must have dimension {self.dimension}.")
        faiss.normalize_L2(query)
        scores, identifiers = self.index.search(np.ascontiguousarray(query), min(top_k, self.size))
        results: list[dict[str, Any]] = []
        for score, identifier in zip(scores[0], identifiers[0]):
            if identifier < 0 or identifier >= len(self.metadata):
                continue
            results.append({**self.metadata[int(identifier)], "score": float(score)})
        return results

    def save(self, index_path: str | Path, metadata_path: str | Path) -> None:
        """Persist the FAISS vectors and aligned metadata to separate local files."""
        if self.index is None:
            raise VectorStoreError("Cannot save an index before creating it.")
        if self.size != len(self.metadata):
            raise VectorStoreError("FAISS vector count does not match metadata count.")
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)
        index_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        _faiss_module().write_index(self.index, str(index_file))
        metadata_file.write_text(json.dumps(self.metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, index_path: str | Path, metadata_path: str | Path) -> "FaissVectorStore":
        """Load a FAISS index and its separate metadata, validating alignment."""
        index_file = Path(index_path)
        metadata_file = Path(metadata_path)
        if not index_file.is_file() or not metadata_file.is_file():
            raise FileNotFoundError("FAISS index or metadata file does not exist.")
        try:
            store = cls()
            store.index = _faiss_module().read_index(str(index_file))
            decoded = json.loads(metadata_file.read_text(encoding="utf-8"))
            if not isinstance(decoded, list) or not all(isinstance(item, dict) for item in decoded):
                raise ValueError("Metadata JSON must contain a list of records.")
            store.metadata = decoded
            if store.size != len(store.metadata):
                raise ValueError("FAISS vector count does not match metadata count.")
            return store
        except Exception as error:
            if isinstance(error, FileNotFoundError):
                raise
            raise VectorStoreError(f"Could not load FAISS index and metadata: {error}") from error