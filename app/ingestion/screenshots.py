"""Utilities for preparing and OCR-ing saved screenshots."""

import os
from datetime import datetime
from pathlib import Path

from .ocr import (
    OCRProcessingError,
    TesseractUnavailableError,
    ensure_tesseract_available,
    extract_text_from_image,
)

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


def scan_screenshots(directory: str | Path) -> list[dict[str, str | None]]:
    """Return metadata for supported image files directly inside a directory.

    Each result includes filename, absolute and directory-relative paths, and
    available creation and modification timestamps in local ISO format.
    """
    screenshot_dir = Path(directory).expanduser()
    if not screenshot_dir.is_dir():
        return []

    screenshots: list[dict[str, str | None]] = []
    for path in sorted(screenshot_dir.iterdir(), key=lambda item: item.name.casefold()):
        if not path.is_file() or path.suffix.casefold() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        try:
            stat = path.stat()
        except OSError:
            continue

        birth_time = getattr(stat, "st_birthtime", None)
        if birth_time is None and os.name == "nt":
            birth_time = stat.st_ctime

        created_at: str | None = None
        modified_at: str | None = None
        if birth_time is not None:
            try:
                created_at = datetime.fromtimestamp(birth_time).astimezone().isoformat(timespec="seconds")
            except (OSError, OverflowError, ValueError):
                pass
        try:
            modified_at = datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds")
        except (OSError, OverflowError, ValueError):
            pass

        screenshots.append(
            {
                "file_name": path.name,
                "absolute_path": str(path.resolve()),
                "relative_path": path.name,
                "created_at": created_at,
                "modified_at": modified_at,
            }
        )
    return screenshots


def ingest_screenshots(directory: str | Path | None = None) -> list[dict[str, str | None]]:
    """OCR supported screenshots and return their metadata and extracted text.

    Empty or missing directories return an empty list. For a non-empty image
    collection, verify the OCR engine before processing starts. A per-image
    processing error is recorded on that image without stopping ingestion;
    engine configuration errors are raised for the whole collection.
    """
    screenshot_dir = (
        Path(directory).expanduser()
        if directory is not None
        else Path(__file__).resolve().parents[2] / "data" / "screenshots"
    )
    screenshots = scan_screenshots(screenshot_dir)
    if not screenshots:
        return []

    ensure_tesseract_available()
    records: list[dict[str, str | None]] = []
    for metadata in screenshots:
        record: dict[str, str | None] = {
            "filename": metadata["file_name"],
            "path": metadata["absolute_path"],
            "created_at": metadata["created_at"],
            "modified_at": metadata["modified_at"],
            "text": "",
        }
        try:
            record["text"] = extract_text_from_image(str(record["path"]))
        except TesseractUnavailableError:
            raise
        except OCRProcessingError as error:
            record["ocr_error"] = str(error)
        records.append(record)
    return records

