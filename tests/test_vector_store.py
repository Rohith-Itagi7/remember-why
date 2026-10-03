"""Tests for the persistent local FAISS vector store and index builder."""

import json
import tempfile
import unittest
from pathlib import Path

from app.retrieval.indexer import build_screenshot_index
from app.retrieval.search import search_index
from app.retrieval.vector_store import FaissVectorStore, VectorStoreError


class FakeEmbedder:
    """Deterministic small vectors without loading the Sentence Transformer."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        vectors = {
            "MCP connects AI apps to tools": [1.0, 0.0],
            "Agents use tools": [0.8, 0.2],
            "A dessert recipe": [0.0, 1.0],
            "Updated MCP protocol": [0.95, 0.05],
        }
        return [vectors[text] for text in texts]

    def embed_text(self, text: str) -> list[float]:
        return [1.0, 0.0] if "MCP" in text or "mcp" in text.lower() else [0.0, 1.0]


class FaissVectorStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)
        self.index_path = self.directory / "screenshots.faiss"
        self.metadata_path = self.directory / "metadata.json"
        self.store = FaissVectorStore()
        self.store.create_index(2)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    @staticmethod
    def metadata(filename: str, text: str) -> dict[str, str | None]:
        return {
            "filename": filename,
            "path": f"/screenshots/{filename}",
            "created_at": "2026-08-01T10:00:00+00:00",
            "modified_at": "2026-08-01T10:00:00+00:00",
            "text": text,
        }

    def test_create_add_and_search_vectors(self) -> None:
        self.store.add_embeddings(
            [[1.0, 0.0], [0.8, 0.2]],
            [self.metadata("mcp.png", "MCP connects AI apps to tools"), self.metadata("agent.png", "Agents use tools")],
        )
        results = self.store.search_embeddings([1.0, 0.0], top_k=2)
        self.assertEqual([result["filename"] for result in results], ["mcp.png", "agent.png"])
        self.assertAlmostEqual(results[0]["score"], 1.0)
        self.assertGreater(results[0]["score"], results[1]["score"])

    def test_save_load_and_metadata_persistence(self) -> None:
        metadata = self.metadata("mcp.png", "MCP connects AI apps to tools")
        self.store.add_embeddings([[1.0, 0.0]], [metadata])
        self.store.save(self.index_path, self.metadata_path)
        loaded = FaissVectorStore.load(self.index_path, self.metadata_path)
        self.assertEqual(loaded.size, 1)
        self.assertEqual(loaded.metadata, [metadata])
        self.assertEqual(loaded.search_embeddings([1.0, 0.0])[0]["text"], metadata["text"])
        self.assertEqual(json.loads(self.metadata_path.read_text(encoding="utf-8")), [metadata])

    def test_empty_index_and_top_k_larger_than_index(self) -> None:
        self.assertEqual(self.store.search_embeddings([1.0, 0.0]), [])
        self.store.add_embeddings([[1.0, 0.0]], [self.metadata("mcp.png", "MCP connects AI apps to tools")])
        self.assertEqual(len(self.store.search_embeddings([1.0, 0.0], top_k=50)), 1)
        self.assertEqual(self.store.search_embeddings([1.0, 0.0], top_k=0), [])

    def test_missing_and_corrupted_files_raise_clear_store_error(self) -> None:
        with self.assertRaises(FileNotFoundError):
            FaissVectorStore.load(self.index_path, self.metadata_path)
        self.index_path.write_bytes(b"not a faiss index")
        self.metadata_path.write_text("[]", encoding="utf-8")
        with self.assertRaises(VectorStoreError):
            FaissVectorStore.load(self.index_path, self.metadata_path)

    def test_duplicate_path_updates_existing_vector_without_adding_duplicate(self) -> None:
        original = self.metadata("mcp.png", "MCP connects AI apps to tools")
        updated = {**original, "text": "Updated MCP protocol"}
        self.store.add_embeddings([[1.0, 0.0]], [original])
        self.store.add_embeddings([[0.95, 0.05]], [updated])
        self.assertEqual(self.store.size, 1)
        self.assertEqual(self.store.metadata[0]["text"], "Updated MCP protocol")
        self.assertEqual(self.store.search_embeddings([1.0, 0.0])[0]["filename"], "mcp.png")


class IndexerTests(unittest.TestCase):
    def test_build_skips_empty_text_and_second_build_reuses_embeddings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            images = root / "screenshots"
            index_directory = root / "index"
            images.mkdir()
            records = [
                {"filename": "mcp.png", "path": str(images / "mcp.png"), "modified_at": "1", "text": "MCP connects AI apps to tools"},
                {"filename": "empty.png", "path": str(images / "empty.png"), "modified_at": "1", "text": "  "},
            ]
            embedder = FakeEmbedder()
            from unittest.mock import patch

            with patch("app.retrieval.indexer.ingest_screenshots", return_value=records):
                first = build_screenshot_index(images, index_directory, embedder)
                second = build_screenshot_index(images, index_directory, embedder)
            self.assertEqual((first["found"], first["extracted"], first["skipped"]), (2, 1, 1))
            self.assertEqual(first["indexed"], 1)
            self.assertEqual(second["changed"], 0)
            self.assertEqual(len(embedder.calls), 1)
            results = search_index("MCP", index_directory=index_directory, embedder=embedder)
            self.assertEqual(results[0]["filename"], "mcp.png")

    def test_build_empty_directory_creates_no_unusable_index(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            empty = root / "screenshots"
            empty.mkdir()
            report = build_screenshot_index(empty, root / "index", FakeEmbedder())
            self.assertEqual(report["found"], 0)
            self.assertFalse(report["index_created"])
            self.assertEqual(search_index("MCP", index_directory=root / "index", embedder=FakeEmbedder()), [])


if __name__ == "__main__":
    unittest.main()