"""MCP client session and narrow LangChain wrappers for Remember Why tools."""

from __future__ import annotations

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool
from mcp import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

TOOL_NAMES = {
    "search_screenshots",
    "get_screenshot_memory",
    "get_memory_context",
    "search_memories_by_date",
}


def _tool_payload(result: Any) -> str:
    """Serialize standard MCP tool output for LangGraph's ToolMessage."""
    if getattr(result, "isError", False):
        messages = [getattr(block, "text", "") for block in getattr(result, "content", [])]
        raise RuntimeError("MCP tool failed: " + " ".join(str(item) for item in messages if item))
    payload = getattr(result, "structured_content", None)
    if isinstance(payload, dict) and "result" in payload:
        payload = payload["result"]
    if payload is None:
        text_parts = [getattr(block, "text", "") for block in getattr(result, "content", [])]
        combined = "\n".join(str(part) for part in text_parts if part)
        try:
            payload = json.loads(combined) if combined else None
        except ValueError:
            payload = combined
    return json.dumps(payload, ensure_ascii=False)


def create_mcp_langchain_tools(session: ClientSession) -> list[StructuredTool]:
    """Wrap only the four registered Remember Why memory tools."""
    async def search_screenshots(query: str, top_k: int = 5) -> str:
        """Search screenshot OCR memories semantically."""
        result = await session.call_tool("search_screenshots", {"query": query, "top_k": top_k})
        return _tool_payload(result)

    async def get_screenshot_memory(filename: str) -> str:
        """Look up metadata for an indexed screenshot filename."""
        result = await session.call_tool("get_screenshot_memory", {"filename": filename})
        return _tool_payload(result)

    async def get_memory_context(filename: str, window_days: int = 7) -> str:
        """Find timestamp-near and semantically related evidence for a screenshot."""
        result = await session.call_tool(
            "get_memory_context",
            {"filename": filename, "window_days": window_days},
        )
        return _tool_payload(result)

    async def search_memories_by_date(date: str, query: str | None = None) -> str:
        """Filter stored screenshot memories by date and optional semantic query."""
        arguments: dict[str, Any] = {"date": date}
        if query is not None:
            arguments["query"] = query
        result = await session.call_tool("search_memories_by_date", arguments)
        return _tool_payload(result)

    return [
        StructuredTool.from_function(
            coroutine=search_screenshots,
            name="search_screenshots",
            description="Search the user's indexed screenshot memories by natural-language meaning.",
        ),
        StructuredTool.from_function(
            coroutine=get_screenshot_memory,
            name="get_screenshot_memory",
            description="Retrieve stored metadata and OCR text by indexed screenshot filename.",
        ),
        StructuredTool.from_function(
            coroutine=get_memory_context,
            name="get_memory_context",
            description=(
                "For questions about why a screenshot may have been saved or what the user was "
                "looking at, return the screenshot plus timestamp-near and semantically related "
                "evidence. This returns evidence only and cannot establish the user's intent."
            ),
        ),
        StructuredTool.from_function(
            coroutine=search_memories_by_date,
            name="search_memories_by_date",
            description="Filter indexed screenshot memories by ISO date, optionally with a semantic query.",
        ),
    ]


@asynccontextmanager
async def remember_why_session():
    """Launch the existing Remember Why server using the active Python interpreter."""
    project_root = Path(__file__).resolve().parents[2]
    parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "app.mcp.server"],
        cwd=str(project_root),
    )
    async with stdio_client(parameters) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            advertised = await session.list_tools()
            available = {tool.name for tool in advertised.tools}
            missing = TOOL_NAMES - available
            if missing:
                raise RuntimeError(f"Remember Why MCP server is missing expected tools: {sorted(missing)}")
            yield session

