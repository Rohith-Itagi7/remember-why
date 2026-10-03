"""Streamlit presentation layer for the existing Remember Why agent."""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any, Callable

from app.agent.graph import context_question_request


_FILENAME_LINE = re.compile(r"^\s*(?:\d+\.\s+)?([^:]+?\.(?:png|jpe?g|webp))\s*$", re.IGNORECASE)
_FIELD_LINE = re.compile(
    r"^\s{2,}(Timestamp|Similarity|Semantic similarity|Time distance|Path|OCR evidence):\s*(.*)$",
    re.IGNORECASE,
)
_SECTION_HEADINGS = (
    "Target memory (direct evidence):",
    "Related memories:",
    "Retrieved evidence:",
    "Possible context:",
)



_CONTEXT_STEM = re.compile(
    r"\b(?:save|saved|keep|kept|surround\w*|around)\s+(?P<stem>[A-Za-z0-9][A-Za-z0-9_.-]*)(?=\s*[?.!,]|$)",
    re.IGNORECASE,
)
_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp")
_GENERIC_REFERENCES = {"this", "it", "that", "something"}


def existing_screenshot_path(path: object) -> str | None:
    """Return a screenshot path only when it points to a readable local file."""
    if not isinstance(path, str) or not path:
        return None
    try:
        candidate = Path(path)
        return str(candidate) if candidate.is_file() else None
    except (OSError, ValueError):
        return None


async def _lookup_indexed_candidate_filenames(candidates: list[str]) -> str | None:
    """Resolve candidate names through the existing safe MCP metadata lookup."""
    from app.agent.mcp_tools import create_mcp_langchain_tools, remember_why_session

    async with remember_why_session() as session:
        lookup_tool = next(
            tool for tool in create_mcp_langchain_tools(session)
            if tool.name == "get_screenshot_memory"
        )
        for candidate in candidates:
            payload = json.loads(await lookup_tool.ainvoke({"filename": candidate}))
            if isinstance(payload, dict):
                indexed_filename = payload.get("filename")
                if isinstance(indexed_filename, str) and indexed_filename:
                    return indexed_filename
    return None


def _resolve_extensionless_context_question(
    question: str,
    screenshot_lookup: Callable[[str], str | None] | None = None,
) -> tuple[str, str | None]:
    """Append an indexed image extension for explicit context-question filename stems."""
    is_context_question, explicit_filename = context_question_request(question)
    if not is_context_question or explicit_filename:
        return question, None

    stem_match = _CONTEXT_STEM.search(question)
    if not stem_match:
        return question, None
    stem = stem_match.group("stem").strip(" .")
    if not stem or stem.casefold() in _GENERIC_REFERENCES:
        return question, None

    candidates = [stem + extension for extension in _IMAGE_EXTENSIONS]
    if screenshot_lookup is None:
        indexed_filename = asyncio.run(_lookup_indexed_candidate_filenames(candidates))
    else:
        indexed_filename = next(
            (found for candidate in candidates if (found := screenshot_lookup(candidate))),
            None,
        )
    if not indexed_filename:
        return question, f"I couldn't find an indexed screenshot matching {stem!r}."

    start, end = stem_match.span("stem")
    return question[:start] + indexed_filename + question[end:], None

def _parse_memory_entries(section: str) -> list[dict[str, Any]]:
    """Parse the agent's existing evidence format into display-only memory records."""
    records: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    reading_ocr = False

    def save_current() -> None:
        if current is not None:
            records.append(current.copy())

    for line in section.splitlines():
        if line.strip() in _SECTION_HEADINGS or line.startswith("Context evidence for "):
            continue
        filename_match = _FILENAME_LINE.match(line)
        if filename_match:
            save_current()
            current = {"filename": filename_match.group(1).strip()}
            reading_ocr = False
            continue
        if current is None:
            continue
        field_match = _FIELD_LINE.match(line)
        if field_match:
            field_name = field_match.group(1).casefold()
            value = field_match.group(2).strip()
            if field_name == "timestamp":
                current["timestamp"] = value
            elif field_name in {"similarity", "semantic similarity"}:
                current["similarity"] = value
            elif field_name == "time distance":
                current["time_distance"] = value
            elif field_name == "path":
                current["path"] = value
            else:
                current["ocr"] = value
                reading_ocr = True
            continue
        if reading_ocr and line.strip():
            current["ocr"] = f"{current.get('ocr', '')}\n{line.rstrip()}".strip()
    save_current()
    return records


def parse_agent_response(response: str) -> dict[str, Any]:
    """Split a grounded agent response into display sections without changing claims."""
    if response.startswith("Context evidence for "):
        target_start = response.find("Target memory (direct evidence):")
        related_start = response.find("Related memories:")
        possible_start = response.find("Possible context:")
        target_text = response[target_start:related_start] if target_start >= 0 and related_start >= 0 else ""
        related_text = response[related_start:possible_start] if related_start >= 0 and possible_start >= 0 else ""
        possible = response[possible_start + len("Possible context:"):].strip() if possible_start >= 0 else ""
        return {
            "kind": "context",
            "target": _parse_memory_entries(target_text),
            "related": _parse_memory_entries(related_text),
            "possible_context": possible,
        }
    if "Retrieved evidence:" in response:
        evidence_text = response.split("Retrieved evidence:", 1)[1]
        return {"kind": "search", "memories": _parse_memory_entries(evidence_text)}
    return {"kind": "message", "message": response}


def _friendly_message(message: str) -> tuple[str, str]:
    """Map backend status text to a short user-facing level and message."""
    lowered = message.casefold()
    detail = lowered.split("details:", 1)[-1] if "details:" in lowered else lowered
    if "couldn't find an indexed screenshot" in lowered:
        return "info", message
    if "couldn't find any indexed memories" in lowered:
        return "info", "I couldn't find any matching memories. Try another question or build the screenshot index."
    if "which screenshot do you mean" in lowered:
        return "info", message
    if "faiss" in detail or "vector store" in detail or "index file" in detail:
        return "error", "The local memory index isn't available. Build it with python main.py --build, then try again."
    if "mcp" in detail or "tool failed" in detail:
        return "error", "Remember Why couldn't reach its local memory tools. Check the MCP service and local index, then try again."
    if "ollama" in detail or "11434" in detail or "connection refused" in detail:
        return "error", "Ollama isn't available right now. Start Ollama and try again."
    if "could not complete" in lowered or "mcp" in lowered:
        return "error", "Remember Why couldn't complete that request. Check the local agent and memory services, then try again."
    return "info", message

def answer_question(
    question: str,
    ask_function: Callable[[str], str] | None = None,
    screenshot_lookup: Callable[[str], str | None] | None = None,
) -> dict[str, Any]:
    """Resolve a context filename stem, then call the existing agent."""
    if not question.strip():
        return {"kind": "message", "level": "info", "message": "Please enter a question about your memories."}

    try:
        question, resolution_error = _resolve_extensionless_context_question(question, screenshot_lookup)
    except Exception as error:
        level, message = _friendly_message(f"Details: {error}")
        if level != "error":
            message = "Remember Why couldn't check the local screenshot index. Check the MCP service and try again."
        return {"kind": "message", "level": "error", "message": message}
    if resolution_error:
        return {"kind": "message", "level": "info", "message": resolution_error}

    try:
        if ask_function is None:
            from app.agent.agent import ask_memory_agent
            ask_function = ask_memory_agent
        response = ask_function(question)
    except Exception as error:
        level, message = _friendly_message(str(error))
        return {"kind": "message", "level": level, "message": message}

    parsed = parse_agent_response(response)
    if parsed["kind"] == "message":
        level, message = _friendly_message(parsed["message"])
        return {"kind": "message", "level": level, "message": message}
    return parsed
