"""Build and update the persistent screenshot embedding index."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Protocol

from app.ingestion.screenshots import ingest_screenshots

from .embeddings import SentenceTransformerEmbeddings
from .vector_store import FaissVectorStore, VectorStoreError


class EmbeddingProvider(Protocol):
    """Embedding interface used by the index builder."""

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Convert texts to vectors."""
        ...


class BuildReport(dict[str, int | bool]):
    """Simple counts describing the last screenshot index build."""


def _default_index_directory() -> Path:
    """Return this project's local data/index directory."""
    return Path(__file__).resolve().parents[2] / "data" / "index"


def _metadata_for(record: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only screenshot fields stored beside a vector."""
    return {
        "filename": record.get("filename") or record.get("file_name") or "",
        "path": record.get("path") or record.get("absolute_path") or "",
        "created_at": record.get("created_at"),
        "modified_at": record.get("modified_at"),
        "text": str(record.get("text") or ""),
    }


def _same_record(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Check whether OCR content and filesystem metadata are unchanged."""
    return all(left.get(key) == right.get(key) for key in ("filename", "path", "created_at", "modified_at", "text"))


def build_screenshot_index(
    screenshot_directory: str | Path | None = None,
    index_directory: str | Path | None = None,
    embedder: EmbeddingProvider | None = None,
) -> BuildReport:
    """OCR screenshots, embed changed readable text, and save/update local FAISS files."""
    root = Path(__file__).resolve().parents[2]
    screenshots_dir = Path(screenshot_directory) if screenshot_directory is not None else root / "data" / "screenshots"
    index_dir = Path(index_directory) if index_directory is not None else _default_index_directory()
    index_path = index_dir / "screenshots.faiss"
    metadata_path = index_dir / "metadata.json"
    records = ingest_screenshots(screenshots_dir)
    searchable = [record for record in records if isinstance(record.get("text"), str) and record["text"].strip()]
    current_metadata = [_metadata_for(record) for record in searchable]

    store: FaissVectorStore | None = None
    if index_path.is_file() and metadata_path.is_file():
        try:
            store = FaissVectorStore.load(index_path, metadata_path)
        except (VectorStoreError, OSError, ValueError):
            store = None
    if store is None:
        store = FaissVectorStore()

    existing = {str(item.get("path") or "").casefold(): item for item in store.metadata}
    paths_to_keep = {str(item.get("path") or "").casefold() for item in current_metadata}
    store.retain_paths(paths_to_keep)
    existing = {str(item.get("path") or "").casefold(): item for item in store.metadata}

    changed_records: list[dict[str, Any]] = []
    for item in current_metadata:
        key = str(item.get("path") or "").casefold()
        if key not in existing or not _same_record(existing[key], item):
            changed_records.append(item)

    if changed_records:
        actual_embedder = embedder or SentenceTransformerEmbeddings()
        vectors = actual_embedder.embed_texts([item["text"] for item in changed_records])
        store.add_embeddings(vectors, changed_records)

    if store.index is not None:
        store.save(index_path, metadata_path)

    ocr_failures = sum(bool(record.get("ocr_error")) for record in records)
    skipped = len(records) - len(searchable)
    return BuildReport(
        found=len(records),
        extracted=len(searchable),
        skipped=skipped,
        ocr_failures=ocr_failures,
        no_readable_text=skipped - ocr_failures,
        indexed=store.size,
        changed=len(changed_records),
        index_created=store.index is not None,
    )

