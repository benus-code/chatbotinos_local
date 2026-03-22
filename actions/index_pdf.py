"""Manual PDF indexation script.

Usage (inside the container or with QDRANT_HOST set):
    python index_pdf.py <path/to/document.pdf> [--no-translate]

FR: Script d'indexation manuelle d'un PDF dans Qdrant.
RU: Скрипт для ручной индексации PDF в Qdrant.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from indexation import (
    build_qdrant_points_pdf,
    client,
    collection_name,
    model,
    replace_pdf_source_points,
)
from pdf_chunker import extract_and_chunk_pdf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a PDF file into Qdrant.")
    parser.add_argument("pdf_path", help="Path to the PDF file to index.")
    parser.add_argument(
        "--no-translate",
        action="store_true",
        help="Skip Russian→French translation (use raw text).",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf_path)
    if not pdf_path.exists():
        LOGGER.error("File not found: %s", pdf_path)
        sys.exit(1)

    LOGGER.info("Extracting and chunking: %s", pdf_path.name)
    chunks = extract_and_chunk_pdf(str(pdf_path), translate=not args.no_translate)
    LOGGER.info("%d chunks produced.", len(chunks))

    if not chunks:
        LOGGER.warning("No chunks extracted — check the PDF content.")
        sys.exit(1)

    LOGGER.info("Generating embeddings and building Qdrant points...")
    points = build_qdrant_points_pdf(chunks, model)

    source_name = pdf_path.name
    LOGGER.info("Replacing existing points for source '%s' in collection '%s'...", source_name, collection_name)
    replace_pdf_source_points(client, collection_name, source_name, points)

    LOGGER.info("Done. %d points indexed for '%s'.", len(points), source_name)


if __name__ == "__main__":
    main()
