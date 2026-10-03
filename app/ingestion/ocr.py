"""Local OCR helpers backed by the Tesseract executable."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

try:
    import pytesseract
except ImportError:
    pytesseract: Any = None


WINDOWS_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


class TesseractUnavailableError(RuntimeError):
    """Raised when pytesseract or the Tesseract executable is unavailable."""


class OCRProcessingError(RuntimeError):
    """Raised when Tesseract cannot process an individual image."""


def find_tesseract_executable() -> str | None:
    """Find Tesseract via a valid override, common Windows paths, then PATH."""
    configured_command = os.environ.get("TESSERACT_CMD", "").strip().strip('"')
    if configured_command:
        configured_path = Path(configured_command).expanduser()
        if configured_path.is_file():
            return str(configured_path)

    for candidate in WINDOWS_TESSERACT_PATHS:
        if Path(candidate).is_file():
            return candidate

    return shutil.which("tesseract")


def ensure_tesseract_available() -> str:
    """Verify the wrapper and local OCR engine, then configure pytesseract.

    Raises an actionable error when the engine is absent. The app never
    downloads or installs the Tesseract executable automatically.
    """
    if pytesseract is None:
        raise TesseractUnavailableError(
            "The pytesseract Python package is not installed. Activate the project "
            "virtual environment and run: python -m pip install -r requirements.txt"
        )

    executable = find_tesseract_executable()
    if executable is None:
        raise TesseractUnavailableError(
            "The pytesseract Python wrapper is installed, but the Tesseract OCR "
            "engine executable is missing or not discoverable. Install Tesseract "
            "OCR separately; Windows normally installs it at "
            r"C:\Program Files\Tesseract-OCR\tesseract.exe or "
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe. "
            "If it is installed elsewhere, set TESSERACT_CMD to the full "
            "tesseract.exe path, or add its folder to PATH."
        )

    pytesseract.pytesseract.tesseract_cmd = executable
    return executable


def extract_text_from_image(image_path: str | Path) -> str:
    """Extract local OCR text from an image, returning an empty string if blank.

    Missing engine configuration raises TesseractUnavailableError.
    Failures processing this particular image raise OCRProcessingError.
    """
    ensure_tesseract_available()
    try:
        return pytesseract.image_to_string(str(image_path)).strip()
    except Exception as error:
        raise OCRProcessingError(f"Tesseract could not process {image_path}: {error}") from error

