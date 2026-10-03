"""Tests for the narrowly scoped Remember Why MCP tools."""

import asyncio
import unittest
from unittest.mock import patch

from app.mcp import server


class MCPServerTests(unittest.TestCase):
    def test_server_initializes_with_only_the_three_memory_tools(self) -> None:
        tools = asyncio.run(server.mcp.list_tools())
        names = {tool.name for tool in tools}
        self.assertEqual(
            names,
            {"search_screenshots", "get_screenshot_memory", "search_memories_by_date", "get_memory_context"},
        )
        self.assertEqual(server.SERVER_NAME, "remember-why")

    def test_search_screenshots_returns_retrieval_fields_and_limits_top_k(self) -> None:
        results = [
            {"filename": "mcp.png", "path": "local/mcp.png", "timestamp": "2026-08-15T10:00:00", "text": "MCP connects apps", "score": 0.91},
        ]
        with patch.object(server, "persistent_search_index", return_value=results) as search:
            returned = server.search_screenshots("What did I save about MCP?", top_k=100)
        search.assert_called_once_with("What did I save about MCP?", top_k=20)
        self.assertEqual(returned, results)

    def test_sdk_dispatches_valid_search_tool_call(self) -> None:
        expected = [{
            "filename": "mcp.png",
            "path": "local/mcp.png",
            "timestamp": "2026-08-15T10:00:00",
            "text": "MCP connects apps",
            "score": 0.91,
        }]
        with patch.object(server, "persistent_search_index", return_value=expected):
            result = asyncio.run(
                server.mcp.call_tool(
                    "search_screenshots",
                    {"query": "What did I save about MCP?", "top_k": 5},
                )
            )
        self.assertEqual(result.structured_content, {"result": expected})

    def test_search_screenshots_clamps_nonpositive_top_k(self) -> None:
        with patch.object(server, "persistent_search_index", return_value=[]) as search:
            self.assertEqual(server.search_screenshots("query", top_k=0), [])
        search.assert_called_once_with("query", top_k=1)

    def test_get_screenshot_memory_returns_stored_metadata(self) -> None:
        stored = [
            {"filename": "mcp.png", "path": "local/mcp.png", "created_at": None, "modified_at": "2026-08-15T10:00:00", "text": "MCP connects apps"},
        ]
        with patch.object(server, "_load_metadata", return_value=stored):
            result = server.get_screenshot_memory("MCP.PNG")
        self.assertEqual(result, {
            "filename": "mcp.png",
            "path": "local/mcp.png",
            "timestamp": "2026-08-15T10:00:00",
            "text": "MCP connects apps",
        })

    def test_missing_filename_returns_none(self) -> None:
        with patch.object(server, "_load_metadata", return_value=[]):
            self.assertIsNone(server.get_screenshot_memory("unknown.png"))

    def test_search_memories_by_date_filters_and_applies_optional_query(self) -> None:
        stored = [
            {"filename": "mcp.png", "path": "p/mcp.png", "created_at": None, "modified_at": "2026-08-15T10:00:00", "text": "MCP connects apps"},
            {"filename": "later.png", "path": "p/later.png", "created_at": None, "modified_at": "2026-08-16T10:00:00", "text": "Other"},
        ]
        ranked = [
            {"filename": "mcp.png", "path": "p/mcp.png", "timestamp": "2026-08-15T10:00:00", "text": "MCP connects apps", "score": 0.91},
            {"filename": "later.png", "path": "p/later.png", "timestamp": "2026-08-16T10:00:00", "text": "Other", "score": 0.8},
        ]
        with patch.object(server, "_load_metadata", return_value=stored), patch.object(server, "persistent_search_index", return_value=ranked) as search:
            result = server.search_memories_by_date("2026-08-15", query="MCP")
        search.assert_called_once_with("MCP", top_k=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["filename"], "mcp.png")
        self.assertEqual(result[0]["score"], 0.91)

    def test_date_search_without_query_only_filters_metadata(self) -> None:
        stored = [
            {"filename": "mcp.png", "path": "p/mcp.png", "modified_at": "2026-08-15T10:00:00", "text": "MCP"},
            {"filename": "later.png", "path": "p/later.png", "modified_at": "2026-08-16T10:00:00", "text": "Later"},
        ]
        with patch.object(server, "_load_metadata", return_value=stored), patch.object(server, "persistent_search_index") as search:
            result = server.search_memories_by_date("2026-08-15")
        search.assert_not_called()
        self.assertEqual([item["filename"] for item in result], ["mcp.png"])

    def test_invalid_date_raises_clear_validation_error(self) -> None:
        with self.assertRaisesRegex(ValueError, "YYYY-MM-DD"):
            server.search_memories_by_date("August 15")


    def test_get_memory_context_returns_target_memory(self) -> None:
        target = {
            "filename": "MCP_Architecture.png",
            "path": "screenshots/MCP_Architecture.png",
            "created_at": "2026-08-15T09:00:00",
            "modified_at": "2026-08-15T10:00:00",
            "text": "MCP servers connect applications to tools.",
        }
        with patch.object(server, "_load_metadata", return_value=[target]), \
             patch.object(server, "persistent_search_index", return_value=[]):
            result = server.get_memory_context("mcp_architecture.PNG")
        self.assertIsNotNone(result)
        self.assertEqual(result["target_memory"], {
            "filename": "MCP_Architecture.png",
            "path": "screenshots/MCP_Architecture.png",
            "timestamp": "2026-08-15T10:00:00",
            "text": "MCP servers connect applications to tools.",
        })
        self.assertEqual(result["window_days"], 7)

    def test_get_memory_context_returns_none_for_unknown_filename(self) -> None:
        with patch.object(server, "_load_metadata", return_value=[]):
            self.assertIsNone(server.get_memory_context("unknown.png"))

    def test_get_memory_context_finds_nearby_records_within_window(self) -> None:
        records = [
            {"filename": "target.png", "path": "p/target.png", "modified_at": "2026-08-15T10:00:00", "text": "Target text"},
            {"filename": "near.png", "path": "p/near.png", "modified_at": "2026-08-12T11:00:00", "text": "Nearby text"},
            {"filename": "far.png", "path": "p/far.png", "modified_at": "2026-08-19T10:00:00", "text": "Far text"},
        ]
        with patch.object(server, "_load_metadata", return_value=records), \
             patch.object(server, "persistent_search_index", return_value=[]):
            result = server.get_memory_context("target.png", window_days=3)
        self.assertEqual([item["filename"] for item in result["nearby_memories"]], ["near.png"])
        self.assertEqual(result["nearby_memories"][0]["time_distance_days"], 2.9583)

    def test_get_memory_context_finds_semantic_records_and_excludes_target(self) -> None:
        records = [
            {"filename": "target.png", "path": "p/target.png", "modified_at": "2026-08-15T10:00:00", "text": "MCP server tool architecture"},
            {"filename": "related.png", "path": "p/related.png", "modified_at": "2026-08-30T10:00:00", "text": "Agents call local tools"},
        ]
        ranked = [
            {"filename": "target.png", "path": "p/target.png", "text": "MCP server tool architecture", "score": 1.0},
            {"filename": "related.png", "path": "p/related.png", "timestamp": "2026-08-30T10:00:00", "text": "Agents call local tools", "score": 0.82},
        ]
        with patch.object(server, "_load_metadata", return_value=records), \
             patch.object(server, "persistent_search_index", return_value=ranked) as search:
            result = server.get_memory_context("target.png")
        search.assert_called_once_with("MCP server tool architecture", top_k=2)
        self.assertEqual([item["filename"] for item in result["semantically_related_memories"]], ["related.png"])
        self.assertEqual(result["semantically_related_memories"][0]["score"], 0.82)

    def test_get_memory_context_handles_no_related_results(self) -> None:
        target = {"filename": "target.png", "path": "p/target.png", "modified_at": "2026-08-15T10:00:00", "text": "Target OCR"}
        with patch.object(server, "_load_metadata", return_value=[target]), \
             patch.object(server, "persistent_search_index") as search:
            result = server.get_memory_context("target.png")
        search.assert_not_called()
        self.assertEqual(result["nearby_memories"], [])
        self.assertEqual(result["semantically_related_memories"], [])

    def test_get_memory_context_ignores_malformed_metadata_timestamps(self) -> None:
        records = [
            None,
            {"filename": "target.png", "path": "p/target.png", "modified_at": "not-a-timestamp", "text": ""},
            {"filename": "broken.png", "path": "p/broken.png", "modified_at": "also-bad", "text": "Some OCR"},
        ]
        with patch.object(server, "_load_metadata", return_value=records), \
             patch.object(server, "persistent_search_index") as search:
            result = server.get_memory_context("target.png")
        search.assert_not_called()
        self.assertEqual(result["target_memory"]["filename"], "target.png")
        self.assertEqual(result["nearby_memories"], [])
        self.assertEqual(result["semantically_related_memories"], [])

if __name__ == "__main__":
    unittest.main()
