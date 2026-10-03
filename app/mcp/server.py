"""Official MCP server exposing narrowly scoped Remember Why memory tools."""

from __future__ import annotations

from datetime import date as Date, datetime
from pathlib import Path
from typing import Any

from mcp.server import MCPServer

from app.retrieval.search import search_index as persistent_search_index
from app.retrieval.vector_store import FaissVectorStore, VectorStoreError

SERVER_NAME = "remember-why"
MAX_TOP_K = 20
DEFAULT_CONTEXT_WINDOW_DAYS = 7
MAX_CONTEXT_RELATED = 5
MAX_CONTEXT_NEARBY = 5

mcp = MCPServer(
    SERVER_NAME,
    version="0.1.0",
    instructions="Search and retrieve screenshot memories from Remember Why's local FAISS index.",
)


def _index_directory() -> Path:
    """Return the fixed project index folder; no tool accepts a path argument."""
    return Path(__file__).resolve().parents[2] / "data" / "index"


def _load_metadata() -> list[dict[str, Any]]:
    """Load metadata already stored beside the local FAISS index."""
    directory = _index_directory()
    try:
        store = FaissVectorStore.load(directory / "screenshots.faiss", directory / "metadata.json")
    except (FileNotFoundError, OSError, VectorStoreError):
        return []
    return store.metadata


def _timestamp(record: dict[str, Any]) -> str | None:
    """Choose the best available stored timestamp for public tool results."""
    value = record.get("modified_at") or record.get("created_at")
    return str(value) if value else None


def _timestamp_epoch(record: dict[str, Any]) -> float | None:
    """Parse stored ISO timestamps safely for comparisons across timezone offsets."""
    value = _timestamp(record)
    if not value:
        return None
    try:
        normalized = value[:-1] + "+00:00" if value.endswith(("Z", "z")) else value
        return datetime.fromisoformat(normalized).timestamp()
    except (OverflowError, OSError, TypeError, ValueError):
        return None


def _public_memory(record: dict[str, Any], *, score: float | None = None) -> dict[str, Any]:
    """Return only documented screenshot memory fields."""
    memory: dict[str, Any] = {
        "filename": record.get("filename", ""),
        "path": record.get("path", ""),
        "timestamp": _timestamp(record),
        "text": record.get("text", ""),
    }
    if score is not None:
        memory["score"] = score
    return memory


def _same_memory(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Compare screenshot identities without confusing distinct same-named files."""
    left_path = str(left.get("path") or "").replace("\\", "/").casefold()
    right_path = str(right.get("path") or "").replace("\\", "/").casefold()
    if left_path and right_path:
        return left_path == right_path
    return str(left.get("filename") or "").casefold() == str(right.get("filename") or "").casefold()


@mcp.tool()
def search_screenshots(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """Search indexed screenshot memories by natural-language meaning."""
    bounded_top_k = max(1, min(int(top_k), MAX_TOP_K))
    results = persistent_search_index(query, top_k=bounded_top_k)
    return [
        {
            "filename": result.get("filename", ""),
            "path": result.get("path", ""),
            "timestamp": result.get("timestamp"),
            "text": result.get("text", ""),
            "score": float(result.get("score", 0.0)),
        }
        for result in results
    ]


@mcp.tool()
def get_screenshot_memory(filename: str) -> dict[str, Any] | None:
    """Return stored metadata for an indexed screenshot filename, if present."""
    target = filename.strip().casefold()
    if not target:
        return None
    for record in _load_metadata():
        if str(record.get("filename", "")).casefold() == target:
            return _public_memory(record)
    return None


@mcp.tool()
def get_memory_context(
    filename: str,
    window_days: int = DEFAULT_CONTEXT_WINDOW_DAYS,
) -> dict[str, Any] | None:
    """Return timestamp-near and semantically related evidence for one screenshot.

    Nearby results use a symmetric time window around the target timestamp.
    Semantic results are ranked by the existing persistent FAISS search using
    the target's OCR text. The target itself is excluded from both collections.
    """
    target_name = filename.strip().casefold()
    if not target_name:
        return None
    try:
        bounded_window = max(0, min(int(window_days), 3650))
    except (TypeError, ValueError, OverflowError):
        raise ValueError("window_days must be a non-negative integer.") from None

    records = [record for record in _load_metadata() if isinstance(record, dict)]
    target = next(
        (record for record in records if str(record.get("filename") or "").casefold() == target_name),
        None,
    )
    if target is None:
        return None

    target_epoch = _timestamp_epoch(target)
    window_seconds = bounded_window * 24 * 60 * 60
    nearby_with_distance: list[tuple[float, dict[str, Any]]] = []
    if target_epoch is not None:
        for record in records:
            if _same_memory(record, target):
                continue
            record_epoch = _timestamp_epoch(record)
            if record_epoch is None:
                continue
            distance = abs(record_epoch - target_epoch)
            if distance <= window_seconds:
                nearby_with_distance.append((distance, record))
    nearby_with_distance.sort(key=lambda item: item[0])
    nearby = [
        {
            **_public_memory(record),
            "time_distance_days": round(distance / (24 * 60 * 60), 4),
        }
        for distance, record in nearby_with_distance[:MAX_CONTEXT_NEARBY]
    ]

    target_text = target.get("text")
    semantic_results: list[dict[str, Any]] = []
    if isinstance(target_text, str) and target_text.strip() and len(records) > 1:
        ranked = persistent_search_index(
            target_text,
            top_k=min(len(records), MAX_CONTEXT_RELATED + 1),
        )
        semantic_results = [
            _public_memory(result, score=float(result.get("score", 0.0)))
            for result in ranked
            if isinstance(result, dict) and not _same_memory(result, target)
        ][:MAX_CONTEXT_RELATED]

    return {
        "target_memory": _public_memory(target),
        "window_days": bounded_window,
        "nearby_memories": nearby,
        "semantically_related_memories": semantic_results,
    }


@mcp.tool()
def search_memories_by_date(date: str, query: str | None = None) -> list[dict[str, Any]]:
    """Filter indexed screenshot memories by ISO date and optionally rank by query."""
    try:
        target_date = Date.fromisoformat(date)
    except (TypeError, ValueError):
        raise ValueError("Invalid date. Use the YYYY-MM-DD format.") from None

    records = _load_metadata()
    matching = [
        record for record in records
        if (_timestamp(record) or "")[:10] == target_date.isoformat()
    ]
    if not query or not query.strip() or not matching:
        return [_public_memory(record) for record in matching]

    ranked = persistent_search_index(query, top_k=max(1, len(records)))
    matching_paths = {str(record.get("path", "")).casefold() for record in matching}
    return [
        {
            "filename": result.get("filename", ""),
            "path": result.get("path", ""),
            "timestamp": result.get("timestamp"),
            "text": result.get("text", ""),
            "score": float(result.get("score", 0.0)),
        }
        for result in ranked
        if str(result.get("path", "")).casefold() in matching_paths
    ]


def main() -> None:
    """Start the MCP server over stdio for a local MCP host to launch."""
    from app.observability.sentry import initialize_sentry

    initialize_sentry()
    mcp.run()


if __name__ == "__main__":
    main()

