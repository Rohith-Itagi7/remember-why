"""Tests for local semantic screenshot search."""

import sys
import unittest
from types import ModuleType
from unittest.mock import Mock, patch

from app.retrieval.embeddings import SentenceTransformerEmbeddings
from app.retrieval.search import search_screenshots


class FakeEmbedder:
    """Deterministic vectors for ranking tests."""

    def __init__(self) -> None:
        self.query_calls: list[str] = []
        self.text_calls: list[list[str]] = []

    def embed_text(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return [1.0, 0.0]

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.text_calls.append(texts)
        vectors = {
            "protocol links ai apps to external tools": [0.9, 0.1],
            "agents can use tools to complete tasks": [0.8, 0.2],
            "a recipe for apple pie": [0.0, 1.0],
        }
        return [vectors[text] for text in texts]


class SemanticSearchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = [
            {
                "filename": "mcp.png",
                "path": "/screenshots/mcp.png",
                "modified_at": "2026-08-14T10:15:00+05:30",
                "text": "protocol links ai apps to external tools",
            },
            {
                "filename": "agent.png",
                "path": "/screenshots/agent.png",
                "created_at": "2026-08-17T12:30:00+05:30",
                "text": "agents can use tools to complete tasks",
            },
            {
                "filename": "recipe.png",
                "path": "/screenshots/recipe.png",
                "text": "a recipe for apple pie",
            },
            {"filename": "blank.png", "text": "  \n "},
        ]
        self.embedder = FakeEmbedder()

    def test_empty_records_return_no_results_without_embedding(self) -> None:
        self.assertEqual(search_screenshots([], "What did I save about MCP?", embedder=self.embedder), [])
        self.assertEqual(self.embedder.query_calls, [])

    def test_empty_ocr_text_is_skipped(self) -> None:
        results = search_screenshots([self.records[-1]], "query", embedder=self.embedder)
        self.assertEqual(results, [])
        self.assertEqual(self.embedder.query_calls, [])

    def test_search_ranks_by_cosine_similarity_and_maps_fields(self) -> None:
        results = search_screenshots(self.records, "What did I save about MCP?", embedder=self.embedder)
        self.assertEqual([result["filename"] for result in results], ["mcp.png", "agent.png", "recipe.png"])
        self.assertGreater(results[0]["score"], results[1]["score"])
        self.assertEqual(results[0]["timestamp"], "2026-08-14T10:15:00+05:30")
        self.assertEqual(results[1]["timestamp"], "2026-08-17T12:30:00+05:30")
        self.assertEqual(results[0]["text"], self.records[0]["text"])
        self.assertEqual(self.embedder.text_calls, [[record["text"] for record in self.records[:3]]])

    def test_top_k_limits_result_count(self) -> None:
        results = search_screenshots(self.records, "query", top_k=2, embedder=self.embedder)
        self.assertEqual(len(results), 2)
        self.assertEqual([item["filename"] for item in results], ["mcp.png", "agent.png"])

    def test_no_searchable_matches_and_nonpositive_top_k_return_empty(self) -> None:
        self.assertEqual(search_screenshots([], "unrelated query", embedder=self.embedder), [])
        self.assertEqual(search_screenshots(self.records, "query", top_k=0, embedder=self.embedder), [])
        self.assertEqual(search_screenshots(self.records, "  ", embedder=self.embedder), [])

    def test_embedding_model_returns_vectors_with_expected_dimensions(self) -> None:
        class FakeModel:
            def encode(self, texts: list[str], **_: object) -> list[list[float]]:
                return [[0.1, 0.2, 0.3] for _ in texts]

        constructor = Mock(return_value=FakeModel())
        fake_module = ModuleType("sentence_transformers")
        fake_module.SentenceTransformer = constructor  # type: ignore[attr-defined]
        with patch.dict(sys.modules, {"sentence_transformers": fake_module}):
            embeddings = SentenceTransformerEmbeddings()
            one = embeddings.embed_text("one")
            many = embeddings.embed_texts(["one", "two"])
        self.assertEqual(len(one), 3)
        self.assertTrue(all(len(vector) == 3 for vector in many))
        self.assertEqual(len(many), 2)
        constructor.assert_called_once_with("all-MiniLM-L6-v2")


if __name__ == "__main__":
    unittest.main()