"""Remember Why local memory agent public API."""


def ask_memory_agent(question: str) -> str:
    """Answer a question using the local LangGraph and Remember Why MCP tools."""
    from .agent import ask_memory_agent as ask

    return ask(question)