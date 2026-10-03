"""Focused tests for Remember Why's Streamlit presentation integration."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from app.agent.agent import ask_memory_agent
from app.ui.presentation import answer_question, existing_screenshot_path, parse_agent_response


class StreamlitPresentationTests(unittest.TestCase):
    def test_screenshot_path_is_returned_only_for_existing_local_files(self) -> None:
        with TemporaryDirectory() as directory:
            screenshot = Path(directory) / "memory.png"
            screenshot.touch()
            self.assertEqual(existing_screenshot_path(str(screenshot)), str(screenshot))
            self.assertIsNone(existing_screenshot_path(str(screenshot.with_name("missing.png"))))
        self.assertIsNone(existing_screenshot_path("bad\0path.png"))
        self.assertIsNone(existing_screenshot_path(None))

    def test_context_response_is_split_into_target_related_and_possible_context(self) -> None:
        response = """Context evidence for MCP_Architecture.png.

Target memory (direct evidence):
MCP_Architecture.png
   Timestamp: 2026-10-02T16:36:39+05:30
   Path: C:/private/MCP_Architecture.png
   OCR evidence: MCP connects applications to tools.

Related memories:
1. Model_Context_Protocol.png
   Timestamp: 2026-10-02T16:36:39+05:30
   Semantic similarity: 0.85
   Time distance: 0.0003 days
   OCR evidence: Model Context Protocol overview.
   Full OCR continuation.

Possible context:
These records may point to topics you were exploring. This is an inference, not proof of intent.
I can't know your exact reason for saving it."""
        parsed = parse_agent_response(response)
        self.assertEqual(parsed["kind"], "context")
        self.assertEqual(parsed["target"][0]["filename"], "MCP_Architecture.png")
        self.assertEqual(parsed["related"][0]["similarity"], "0.85")
        self.assertEqual(parsed["related"][0]["time_distance"], "0.0003 days")
        self.assertIn("Full OCR continuation.", parsed["related"][0]["ocr"])
        self.assertIn("inference", parsed["possible_context"])

    def test_semantic_search_response_keeps_results_for_presentation(self) -> None:
        parsed = parse_agent_response(
            """I found 1 relevant memory.

Retrieved evidence:
1. mcp.png
   Timestamp: 2026-10-02
   Similarity: 0.91
   OCR evidence: MCP allows applications to connect tools."""
        )
        self.assertEqual(parsed["kind"], "search")
        self.assertEqual(parsed["memories"][0]["filename"], "mcp.png")
        self.assertEqual(parsed["memories"][0]["similarity"], "0.91")

    def test_question_uses_existing_agent_entry_point(self) -> None:
        with patch("app.agent.agent.ask_memory_agent", return_value="I couldn't find any indexed memories related to your question.") as ask:
            result = answer_question("What did I save about MCP?")
        ask.assert_called_once_with("What did I save about MCP?")
        self.assertEqual(result["kind"], "message")
        self.assertEqual(result["level"], "info")

    def test_empty_question_does_not_call_agent(self) -> None:
        with patch("app.agent.agent.ask_memory_agent") as ask:
            result = answer_question("  ")
        ask.assert_not_called()
        self.assertEqual(result["message"], "Please enter a question about your memories.")

    def test_faiss_failure_is_not_mislabeled_as_ollama(self) -> None:
        response = (
            "The local memory agent could not complete the request. Ensure Ollama is installed. "
            "Details: VectorStoreError: FAISS index or metadata file does not exist."
        )
        with patch("app.agent.agent.ask_memory_agent", return_value=response):
            result = answer_question("What did I save about MCP?")
        self.assertEqual(result["level"], "error")
        self.assertIn("local memory index", result["message"])
        self.assertNotIn("Ollama", result["message"])

    def test_mcp_failure_is_not_mislabeled_as_ollama(self) -> None:
        response = (
            "The local memory agent could not complete the request. Ensure Ollama is installed. "
            "Details: RuntimeError: MCP tool failed."
        )
        with patch("app.agent.agent.ask_memory_agent", return_value=response):
            result = answer_question("What did I save about MCP?")
        self.assertEqual(result["level"], "error")
        self.assertIn("local memory tools", result["message"])
        self.assertNotIn("Ollama", result["message"])
    def test_extensionless_context_stem_resolves_to_indexed_png(self) -> None:
        looked_up: list[str] = []

        def lookup(candidate: str) -> str | None:
            looked_up.append(candidate)
            return "Agent_tools.png" if candidate == "Agent_tools.png" else None

        ask = Mock(return_value="Context evidence for Agent_tools.png.")
        result = answer_question("Why did I save Agent_tools?", ask, lookup)
        self.assertEqual(looked_up, ["Agent_tools.png"])
        ask.assert_called_once_with("Why did I save Agent_tools.png?")
        self.assertEqual(result["kind"], "context")

    def test_extensionless_filename_in_context_surrounds_question_resolves(self) -> None:
        ask = Mock(return_value="Context evidence for Agent_tools.png.")
        result = answer_question(
            "What context surrounds Agent_tools?",
            ask,
            lambda candidate: "Agent_tools.png" if candidate == "Agent_tools.png" else None,
        )
        ask.assert_called_once_with("What context surrounds Agent_tools.png?")
        self.assertEqual(result["kind"], "context")
    def test_explicit_png_context_filename_is_passed_through(self) -> None:
        lookup = Mock()
        ask = Mock(return_value="Context evidence for Agent_tools.png.")
        result = answer_question("Why did I save Agent_tools.png?", ask, lookup)
        lookup.assert_not_called()
        ask.assert_called_once_with("Why did I save Agent_tools.png?")
        self.assertEqual(result["kind"], "context")

    def test_unknown_extensionless_name_keeps_helpful_not_found_message(self) -> None:
        looked_up: list[str] = []

        def lookup(candidate: str) -> str | None:
            looked_up.append(candidate)
            return None

        ask = Mock()
        result = answer_question("Why did I save Not_in_my_index?", ask, lookup)
        self.assertEqual(looked_up, ["Not_in_my_index.png", "Not_in_my_index.jpg", "Not_in_my_index.jpeg", "Not_in_my_index.webp"])
        ask.assert_not_called()
        self.assertEqual(result["level"], "info")
        self.assertIn("couldn't find an indexed screenshot matching 'Not_in_my_index'", result["message"])

    def test_mcp_lookup_failure_is_shown_without_traceback(self) -> None:
        ask = Mock()
        result = answer_question(
            "Why did I save Agent_tools?",
            ask,
            Mock(side_effect=RuntimeError("MCP server offline")),
        )
        ask.assert_not_called()
        self.assertEqual(result["kind"], "message")
        self.assertEqual(result["level"], "error")
        self.assertIn("couldn't reach its local memory tools", result["message"])
    def test_explicit_mcp_architecture_png_still_uses_existing_context_path(self) -> None:
        lookup = Mock()
        ask = Mock(return_value="Context evidence for MCP_Architecture.png.")
        result = answer_question("Why did I save MCP_Architecture.png?", ask, lookup)
        lookup.assert_not_called()
        ask.assert_called_once_with("Why did I save MCP_Architecture.png?")
        self.assertEqual(result["kind"], "context")
    def test_ollama_failure_is_human_readable(self) -> None:
        with patch("app.agent.agent.ask_memory_agent", side_effect=RuntimeError("Ollama connection refused")):
            result = answer_question("What did I save about MCP?")
        self.assertEqual(result["level"], "error")
        self.assertIn("Ollama isn't available", result["message"])


if __name__ == "__main__":
    unittest.main()
