"""Build, search, or ask the local evidence-grounded memory agent."""

import argparse
from typing import Sequence

from app.ingestion.ocr import TesseractUnavailableError
from app.retrieval.indexer import build_screenshot_index
from app.retrieval.search import search_index


def main(arguments: Sequence[str] | None = None) -> None:
    """Run screenshot index building, direct search, or the local agent."""
    parser = argparse.ArgumentParser(description="Build or search local screenshot memory.")
    operations = parser.add_mutually_exclusive_group(required=True)
    operations.add_argument("--build", action="store_true", help="OCR screenshots and update the FAISS index")
    operations.add_argument("--search", metavar="QUERY", help="Search the saved screenshot index")
    operations.add_argument("--ask", metavar="QUESTION", help="Ask the local evidence-grounded memory agent")
    options = parser.parse_args(arguments)

    if options.build:
        print("Building screenshot memory...\n")
        try:
            report = build_screenshot_index()
        except TesseractUnavailableError as error:
            print(f"Tesseract not installed/configured: {error}")
            return
        except RuntimeError as error:
            print(f"Index build could not complete: {error}")
            return
        print("Tesseract OCR engine available: yes.")
        print(f"Found {report['found']} screenshots.")
        print(f"Extracted text from {report['extracted']} screenshots.")
        print(f"Skipped {report['no_readable_text']} screenshots where Tesseract returned no readable text.")
        if report["ocr_failures"]:
            print(f"OCR processing failed for {report['ocr_failures']} individual screenshot(s).")
        if report["index_created"]:
            print("\nFAISS index created successfully.")
            print(f"Indexed screenshots: {report['indexed']} (updated: {report['changed']})")
        else:
            print("\nNo searchable OCR text was found; no FAISS index was created.")
        return

    if options.ask is not None:
        from app.observability.sentry import initialize_sentry
        initialize_sentry()
        from app.agent.agent import ask_memory_agent

        print(ask_memory_agent(options.ask))
        return

    try:
        results = search_index(options.search)
    except RuntimeError as error:
        print(f"Search could not complete: {error}")
        return
    print("Search results:")
    if not results:
        print("No results. Build an index with: python main.py --build")
        return
    for position, result in enumerate(results, start=1):
        print(f"\n{position}. {result.get('filename', '')}")
        print(f"   Score: {result['score']:.2f}")
        print(f"   Text: {result.get('text', '')}")


if __name__ == "__main__":
    main()

