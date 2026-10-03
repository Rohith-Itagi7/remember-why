"""Local Ollama-backed LangGraph memory agent using the Remember Why MCP server."""

from __future__ import annotations

import asyncio
import os
from typing import Any

from .graph import build_memory_graph, context_question_request, initial_messages
from .mcp_tools import create_mcp_langchain_tools, remember_why_session
from app.observability.sentry import agent_run_span

DEFAULT_MODEL = "qwen3:4b"
LOCAL_OLLAMA_URL = "http://127.0.0.1:11434"

def configured_model_name() -> str:
    """Return the explicitly configured local Ollama model name."""
    return os.environ.get("REMEMBER_WHY_MODEL", DEFAULT_MODEL).strip() or DEFAULT_MODEL


def create_local_model(model_name: str | None = None) -> Any:
    """Create a ChatOllama client without pulling/downloading a model."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as error:
        raise RuntimeError("LangChain Ollama support is missing. Install project requirements in .venv.") from error
    return ChatOllama(
        model=model_name or configured_model_name(),
        base_url=LOCAL_OLLAMA_URL,
        temperature=0,
    )


async def _ask_with_model(question: str, model: Any) -> str:
    """Connect through MCP, invoke the LangGraph workflow, and return its grounded output."""
    async with remember_why_session() as session:
        tools = create_mcp_langchain_tools(session)
        graph = build_memory_graph(model, tools)
        result = await graph.ainvoke(
            {"messages": initial_messages(question)},
            config={"recursion_limit": 8},
        )
    messages = result.get("messages", [])
    if not messages:
        return "I couldn't find any indexed memories related to your question."
    return str(messages[-1].content)


def ask_memory_agent(question: str) -> str:
    """Answer a memory question using local Ollama and evidence from MCP tools.

    No model is downloaded automatically. Final factual output is rendered from
    validated MCP records rather than free-form model claims.
    """
    if not question.strip():
        return "Please enter a question about your indexed memories."
    is_context_question, filename = context_question_request(question)
    if is_context_question and not filename:
        return "Which screenshot do you mean? Please include its indexed filename so I can retrieve context."
    model_name = configured_model_name()
    try:
        model = create_local_model(model_name)
        with agent_run_span():
            return asyncio.run(_ask_with_model(question, model))
    except Exception as error:
        detail = f"{type(error).__name__}: {error}"
        return (
            "The local memory agent could not complete the request. Ensure Ollama is installed and running, "
            f"and that model '{model_name}' is already installed (for example, run ollama pull {model_name} yourself). "
            f"No model was downloaded automatically. Details: {detail}"
        )

