"""Tests for screenshot OCR ingestion without a Tesseract installation."""

import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from app.ingestion import ocr
from app.ingestion.ocr import OCRProcessingError, TesseractUnavailableError
from app.ingestion.screenshots import ingest_screenshots, scan_screenshots


class ScreenshotIngestionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp_dir.name)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def fake_pytesseract(self, image_to_string=None):
        return SimpleNamespace(
            pytesseract=SimpleNamespace(tesseract_cmd="tesseract"),
            image_to_string=image_to_string or unittest.mock.Mock(return_value="recognized text"),
        )

    def test_empty_directory_returns_no_records(self) -> None:
        with patch("app.ingestion.screenshots.ensure_tesseract_available") as check:
            self.assertEqual(ingest_screenshots(self.directory), [])
            self.assertEqual(ingest_screenshots(self.directory / "missing"), [])
        check.assert_not_called()

    def test_unsupported_files_are_ignored(self) -> None:
        (self.directory / "notes.txt").write_text("not an image", encoding="utf-8")
        (self.directory / "image.gif").write_bytes(b"unsupported")
        with patch("app.ingestion.screenshots.ensure_tesseract_available") as check:
            self.assertEqual(scan_screenshots(self.directory), [])
            self.assertEqual(ingest_screenshots(self.directory), [])
        check.assert_not_called()

    def test_tesseract_cmd_valid_file_is_preferred_and_configured(self) -> None:
        executable = self.directory / "custom-tesseract.exe"
        executable.write_bytes(b"mock executable")
        fake = self.fake_pytesseract()
        with patch.dict(os.environ, {"TESSERACT_CMD": str(executable)}), \
             patch.object(ocr, "pytesseract", fake), \
             patch.object(ocr.shutil, "which", return_value="path-tesseract"):
            self.assertEqual(ocr.ensure_tesseract_available(), str(executable))
        self.assertEqual(fake.pytesseract.tesseract_cmd, str(executable))

    def test_standard_windows_install_path_is_discovered(self) -> None:
        fake = self.fake_pytesseract()
        expected = ocr.WINDOWS_TESSERACT_PATHS[0]
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(ocr, "WINDOWS_TESSERACT_PATHS", (expected,)), patch.object(ocr.Path, "is_file", return_value=True), \
             patch.object(ocr.shutil, "which", return_value=None), \
             patch.object(ocr, "pytesseract", fake):
            self.assertEqual(ocr.ensure_tesseract_available(), expected)

    def test_path_based_discovery(self) -> None:
        fake = self.fake_pytesseract()
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(ocr.Path, "is_file", return_value=False), \
             patch.object(ocr.shutil, "which", return_value=r"C:\Tools\tesseract.exe"), \
             patch.object(ocr, "pytesseract", fake):
            self.assertEqual(ocr.ensure_tesseract_available(), r"C:\Tools\tesseract.exe")

    def test_tesseract_unavailable_has_actionable_configuration_error(self) -> None:
        fake = self.fake_pytesseract()
        with patch.dict(os.environ, {}, clear=True), \
             patch.object(ocr.Path, "is_file", return_value=False), \
             patch.object(ocr.shutil, "which", return_value=None), \
             patch.object(ocr, "pytesseract", fake):
            with self.assertRaises(TesseractUnavailableError) as context:
                ocr.ensure_tesseract_available()
        message = str(context.exception)
        self.assertIn("pytesseract Python wrapper is installed", message)
        self.assertIn("Tesseract OCR engine executable is missing", message)
        self.assertIn(r"C:\Program Files\Tesseract-OCR\tesseract.exe", message)
        self.assertIn("TESSERACT_CMD", message)

    def test_ocr_returns_text_or_empty_for_unreadable_image(self) -> None:
        image = self.directory / "sample.png"
        image.write_bytes(b"mock image")
        fake = self.fake_pytesseract(unittest.mock.Mock(side_effect=[" hello\n", "  "]))
        with patch.object(ocr, "pytesseract", fake), \
             patch.object(ocr, "ensure_tesseract_available", return_value="mock-tesseract"):
            self.assertEqual(ocr.extract_text_from_image(image), "hello")
            self.assertEqual(ocr.extract_text_from_image(image), "")

    def test_ocr_wraps_image_processing_failure(self) -> None:
        image = self.directory / "sample.png"
        image.write_bytes(b"mock image")
        fake = self.fake_pytesseract(unittest.mock.Mock(side_effect=RuntimeError("recognition failed")))
        with patch.object(ocr, "pytesseract", fake), \
             patch.object(ocr, "ensure_tesseract_available", return_value="mock-tesseract"):
            with self.assertRaisesRegex(OCRProcessingError, "recognition failed"):
                ocr.extract_text_from_image(image)

    def test_ingestion_checks_availability_before_processing_images(self) -> None:
        image = self.directory / "sample.png"
        image.write_bytes(b"mock image")
        with patch("app.ingestion.screenshots.ensure_tesseract_available",
                   side_effect=TesseractUnavailableError("engine missing")), \
             patch("app.ingestion.screenshots.extract_text_from_image") as extract:
            with self.assertRaises(TesseractUnavailableError):
                ingest_screenshots(self.directory)
        extract.assert_not_called()

    def test_successful_ingestion_preserves_metadata_and_text(self) -> None:
        image = self.directory / "mcp.png"
        image.write_bytes(b"mock image")
        with patch("app.ingestion.screenshots.ensure_tesseract_available"), \
             patch("app.ingestion.screenshots.extract_text_from_image", return_value="MCP connects applications"):
            records = ingest_screenshots(self.directory)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["filename"], "mcp.png")
        self.assertEqual(records[0]["path"], str(image.resolve()))
        self.assertIsNotNone(records[0]["modified_at"])
        self.assertEqual(records[0]["text"], "MCP connects applications")

    def test_failure_for_one_image_does_not_stop_remaining_ingestion(self) -> None:
        (self.directory / "a.png").write_bytes(b"mock image")
        (self.directory / "b.jpg").write_bytes(b"mock image")
        with patch("app.ingestion.screenshots.ensure_tesseract_available"), \
             patch(
                 "app.ingestion.screenshots.extract_text_from_image",
                 side_effect=[OCRProcessingError("bad image"), "second image text"],
             ):
            records = ingest_screenshots(self.directory)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["text"], "")
        self.assertEqual(records[0]["ocr_error"], "bad image")
        self.assertEqual(records[1]["text"], "second image text")


if __name__ == "__main__":
    unittest.main()

