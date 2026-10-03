"""LangGraph memory agent behavior and grounding tests."""

import asyncio
import json
import unittest
from unittest.mock import AsyncMock, patch

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import StructuredTool

from app.agent.agent import ask_memory_agent
from app.agent.graph import NO_EVIDENCE, _grounded_answer, build_memory_graph, initial_messages
from app.agent.mcp_tools import _tool_payload, create_mcp_langchain_tools


class FakeSession:
    def __init__(self, result: object) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, object]]] = []

    async def call_tool(self, name: str, arguments: dict[str, object]) -> object:
        self.calls.append((name, arguments))
        return self.result


class FakeModel:
    """A minimal tool-calling chat model double."""

    def __init__(self, responses: list[AIMessage]) -> None:
        self.responses = responses
        self.calls = 0
        self.bound_tools: list[object] = []

    def bind_tools(self, tools: list[object]) -> "FakeModel":
        self.bound_tools = tools
        return self

    def invoke(self, messages: list[object]) -> AIMessage:
        response = self.responses[self.calls]
        self.calls += 1
        return response


def sample_tool_result(records: object) -> object:
    return type("MCPResult", (), {"structured_content": {"result": records}, "isError": False, "content": []})()


class MemoryAgentTests(unittest.TestCase):
    def test_agent_initialization_builds_langgraph(self) -> None:
        model = FakeModel([AIMessage(content="done")])
        tools = [StructuredTool.from_function(lambda value: value, name="echo", description="test")]
        graph = build_memory_graph(model, tools)
        nodes = set(graph.get_graph().nodes)
        self.assertTrue({"agent", "tools", "grounded_response"}.issubset(nodes))
        self.assertIs(model.bound_tools[0], tools[0])

    def test_mcp_tool_invocation_uses_existing_server_tool(self) -> None:
        result = sample_tool_result([{"filename": "mcp.png", "text": "MCP connects tools"}])
        session = FakeSession(result)
        tools = create_mcp_langchain_tools(session)  # type: ignore[arg-type]
        search_tool = next(tool for tool in tools if tool.name == "search_screenshots")
        output = asyncio.run(search_tool.ainvoke({"query": "MCP", "top_k": 3}))
        self.assertEqual(session.calls, [("search_screenshots", {"query": "MCP", "top_k": 3})])
        self.assertEqual(json.loads(output), [{"filename": "mcp.png", "text": "MCP connects tools"}])

    def test_grounded_response_uses_mocked_mcp_evidence_not_model_claims(self) -> None:
        evidence = {
            "filename": "mcp.png",
            "path": "local/mcp.png",
            "timestamp": "2026-08-15T10:00:00",
            "text": "MCP servers expose tools.",
            "score": 0.91,
        }
        session = FakeSession(sample_tool_result([evidence]))
        tools = create_mcp_langchain_tools(session)  # type: ignore[arg-type]
        model = FakeModel([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "search_screenshots",
                    "args": {"query": "What did I save about MCP?", "top_k": 5},
                    "id": "call-1",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="The user saved this because they are building a startup."),
        ])
        graph = build_memory_graph(model, tools)
        result = asyncio.run(graph.ainvoke({"messages": initial_messages("What did I save about MCP?")}))
        answer = result["messages"][-1].content
        self.assertIn("mcp.png", answer)
        self.assertIn("2026-08-15T10:00:00", answer)
        self.assertIn("MCP servers expose tools.", answer)
        self.assertNotIn("startup", answer)
        self.assertEqual(session.calls, [("search_screenshots", {"query": "What did I save about MCP?", "top_k": 5})])

    def test_no_evidence_returns_clear_no_memory_response(self) -> None:
        answer = _grounded_answer("MCP?", [ToolMessage(content="[]", tool_call_id="empty")])
        self.assertEqual(answer, NO_EVIDENCE)

    def test_malformed_tool_result_is_not_treated_as_evidence(self) -> None:
        malformed = ToolMessage(content="not valid JSON", tool_call_id="bad")
        answer = _grounded_answer("MCP?", [malformed])
        self.assertEqual(answer, NO_EVIDENCE)

    def test_agent_failure_returns_local_ollama_guidance(self) -> None:
        with patch("app.agent.agent.create_local_model", side_effect=RuntimeError("Ollama unavailable")):
            answer = ask_memory_agent("What did I save about MCP?")
        self.assertIn("Ollama is installed and running", answer)
        self.assertIn("No model was downloaded automatically", answer)

    def test_ollama_model_is_configurable(self) -> None:
        from app.agent.agent import configured_model_name

        with patch.dict("os.environ", {}, clear=True):
            self.assertEqual(configured_model_name(), "qwen3:4b")
        with patch.dict("os.environ", {"REMEMBER_WHY_MODEL": "llama3.2:3b"}):
            self.assertEqual(configured_model_name(), "llama3.2:3b")


    def test_mcp_context_tool_invocation_uses_existing_server_tool(self) -> None:
        context = {
            "target_memory": {"filename": "mcp.png", "path": "p/mcp.png", "timestamp": "2026-08-15", "text": "MCP tools"},
            "window_days": 7,
            "nearby_memories": [],
            "semantically_related_memories": [],
        }
        session = FakeSession(sample_tool_result(context))
        tools = create_mcp_langchain_tools(session)  # type: ignore[arg-type]
        context_tool = next(tool for tool in tools if tool.name == "get_memory_context")
        output = asyncio.run(context_tool.ainvoke({"filename": "mcp.png", "window_days": 4}))
        self.assertEqual(session.calls, [("get_memory_context", {"filename": "mcp.png", "window_days": 4})])
        self.assertEqual(json.loads(output), context)

    def test_context_answer_separates_evidence_and_possible_inference(self) -> None:
        from app.agent.graph import _grounded_answer

        context = {
            "target_memory": {
                "filename": "MCP_Architecture.png",
                "path": "screenshots/MCP_Architecture.png",
                "timestamp": "2026-08-15T10:00:00",
                "text": "MCP servers connect apps to tools.",
            },
            "window_days": 7,
            "nearby_memories": [{
                "filename": "agent_tools.png",
                "path": "screenshots/agent_tools.png",
                "timestamp": "2026-08-16T10:00:00",
                "text": "Agents can use tools.",
                "time_distance_days": 1.0,
            }],
            "semantically_related_memories": [{
                "filename": "local_agents.png",
                "path": "screenshots/local_agents.png",
                "timestamp": "2026-09-01T10:00:00",
                "text": "A local agent can call tools.",
                "score": 0.84,
            }],
        }
        message = ToolMessage(content=json.dumps(context), tool_call_id="context-call")
        answer = _grounded_answer("Why did I save MCP_Architecture.png?", [message])
        self.assertIn("Target memory (direct evidence)", answer)
        self.assertIn("Related memories:", answer)
        self.assertNotIn("Semantically related memories:", answer)
        self.assertIn("Possible context:", answer)
        self.assertIn("I can't know your exact reason for saving it", answer)
        self.assertIn("MCP servers connect apps to tools.", answer)
        self.assertNotIn("You saved this because", answer)

    def _context_answer(self, nearby: list[dict[str, object]], semantic: list[dict[str, object]]) -> str:
        context = {
            "target_memory": {
                "filename": "target.png",
                "path": "screenshots/target.png",
                "timestamp": "2026-10-02T12:00:00",
                "text": "Target OCR text.",
            },
            "window_days": 7,
            "nearby_memories": nearby,
            "semantically_related_memories": semantic,
        }
        return _grounded_answer(
            "Why did I save target.png?",
            [ToolMessage(content=json.dumps(context), tool_call_id="dedupe-context")],
        )

    def test_context_related_memory_only_from_nearby_is_included_once(self) -> None:
        nearby = {
            "filename": "nearby.png",
            "path": "screenshots/nearby.png",
            "timestamp": "2026-10-02T12:01:00",
            "text": "Nearby OCR text.",
            "time_distance_days": 0.0007,
        }
        answer = self._context_answer([nearby], [])
        self.assertEqual(answer.count("1. nearby.png"), 1)
        self.assertIn("Time distance: 0.0007 days", answer)
        self.assertIn("Nearby OCR text.", answer)

    def test_context_related_memory_only_from_semantic_is_included_once(self) -> None:
        semantic = {
            "filename": "semantic.png",
            "path": "screenshots/semantic.png",
            "timestamp": "2026-10-01T12:00:00",
            "text": "Semantic OCR text.",
            "score": 0.85,
        }
        answer = self._context_answer([], [semantic])
        self.assertEqual(answer.count("1. semantic.png"), 1)
        self.assertIn("Semantic similarity: 0.85", answer)
        self.assertIn("Semantic OCR text.", answer)

    def test_context_merges_duplicate_and_excludes_target_from_related(self) -> None:
        nearby = {
            "filename": "Model_Context_Protocol.png",
            "path": "screenshots/Model_Context_Protocol.png",
            "timestamp": "2026-10-02T16:36:39+05:30",
            "text": "MCP timestamp-near OCR.",
            "time_distance_days": 0.0003,
        }
        semantic = {
            "filename": "model_context_protocol.PNG",
            "path": "C:/memory/screenshots/Model_Context_Protocol.png",
            "timestamp": "2026-10-02T16:36:39+05:30",
            "text": "MCP semantic OCR.",
            "score": 0.85,
        }
        target = {
            "filename": "target.png",
            "path": "screenshots/target.png",
            "timestamp": "2026-10-02T12:00:00",
            "text": "Target OCR text.",
        }
        answer = self._context_answer(
            [nearby, {**target, "time_distance_days": 0.0}],
            [semantic, {**target, "score": 1.0}],
        )
        self.assertEqual(answer.count("Model_Context_Protocol.png"), 1)
        related_section = answer.split("Related memories:", 1)[1].split("Possible context:", 1)[0]
        self.assertNotIn("target.png", related_section)
        self.assertIn("Time distance: 0.0003 days", answer)
        self.assertIn("Semantic similarity: 0.85", answer)
        self.assertIn("MCP semantic OCR.", answer)
        self.assertIn("Possible context:", answer)
        self.assertIn("not proof of your intent", answer)
    def test_what_context_surrounds_filename_routes_to_context_tool(self) -> None:
        context = {
            "target_memory": {
                "filename": "Pytorch.png",
                "path": "p/Pytorch.png",
                "timestamp": "2026-08-15T10:00:00",
                "text": "PyTorch model training.",
            },
            "window_days": 7,
            "nearby_memories": [],
            "semantically_related_memories": [],
        }
        session = FakeSession(sample_tool_result(context))
        tools = create_mcp_langchain_tools(session)  # type: ignore[arg-type]
        model = FakeModel([AIMessage(content="The model should not decide this route.")])
        graph = build_memory_graph(model, tools)
        result = asyncio.run(
            graph.ainvoke({"messages": initial_messages("What context surrounds Pytorch.png?")})
        )
        self.assertEqual(
            session.calls,
            [("get_memory_context", {"filename": "Pytorch.png", "window_days": 7})],
        )
        self.assertEqual(model.calls, 0)
        self.assertIn("Target memory (direct evidence)", result["messages"][-1].content)

    def test_context_question_with_unknown_filename_returns_not_found(self) -> None:
        session = FakeSession(sample_tool_result(None))
        tools = create_mcp_langchain_tools(session)  # type: ignore[arg-type]
        model = FakeModel([AIMessage(content="The model should not decide this route.")])
        graph = build_memory_graph(model, tools)
        result = asyncio.run(
            graph.ainvoke({"messages": initial_messages("Why did I save unknown.png?")})
        )
        self.assertEqual(
            session.calls,
            [("get_memory_context", {"filename": "unknown.png", "window_days": 7})],
        )
        self.assertEqual(model.calls, 0)
        self.assertIn("couldn't find an indexed screenshot named 'unknown.png'", result["messages"][-1].content)
    def test_unanchored_why_question_asks_for_screenshot_filename(self) -> None:
        with patch("app.agent.agent.create_local_model") as create_model:
            answer = ask_memory_agent("Why did I save this?")
        create_model.assert_not_called()
        self.assertIn("Which screenshot do you mean?", answer)

    def test_graph_uses_memory_context_tool_and_ignores_model_reason_claims(self) -> None:
        context = {
            "target_memory": {
                "filename": "MCP_Architecture.png",
                "path": "p/MCP_Architecture.png",
                "timestamp": "2026-08-15T10:00:00",
                "text": "MCP servers connect tools.",
            },
            "window_days": 7,
            "nearby_memories": [],
            "semantically_related_memories": [],
        }
        session = FakeSession(sample_tool_result(context))
        tools = create_mcp_langchain_tools(session)  # type: ignore[arg-type]
        model = FakeModel([
            AIMessage(
                content="",
                tool_calls=[{
                    "name": "get_memory_context",
                    "args": {"filename": "MCP_Architecture.png", "window_days": 7},
                    "id": "context-call",
                    "type": "tool_call",
                }],
            ),
            AIMessage(content="The user saved it because they are launching a company."),
        ])
        graph = build_memory_graph(model, tools)
        result = asyncio.run(
            graph.ainvoke({"messages": initial_messages("Why did I save MCP_Architecture.png?")})
        )
        answer = result["messages"][-1].content
        self.assertEqual(
            session.calls,
            [("get_memory_context", {"filename": "MCP_Architecture.png", "window_days": 7})],
        )
        self.assertIn("Possible context:", answer)
        self.assertIn("MCP servers connect tools.", answer)
        self.assertNotIn("launching a company", answer)


if __name__ == "__main__":
    unittest.main()
