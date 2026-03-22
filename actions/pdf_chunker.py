"""PDF text extraction and chunking strategies for Qdrant indexation.

Two strategies are supported:
- 'legal'  : article-based chunking for Russian regulatory docs (ПОЛОЖЕНИЕ, ПРИКАЗ).
- 'guide'  : token-based sliding-window for prose documents (student brochures).

FR: Extraction et découpage de PDFs selon le type de document.
RU: Извлечение текста и разбивка PDF в зависимости от типа документа.
"""

from __future__ import annotations

import re
import time
from pathlib import Path
from typing import Dict, List, Optional

LEGAL_DOC_PATTERN = re.compile(r"\b(ПРИКАЗ|ПОЛОЖЕНИЕ)\b")
ARTICLE_PATTERN = re.compile(r"^(\d+\.\d+)\.\s+(.+)$")
SECTION_PATTERN = re.compile(r"^(\d+)\.\s+(.+)$")

# Minimum word count to keep a chunk; shorter ones are merged into previous.
_MIN_CHUNK_WORDS = 50


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract raw text from all pages, sorted top-to-bottom per page.

    FR: Extrait le texte brut du PDF, trié spatialement par page.
    RU: Извлекает текст из PDF, отсортированный пространственно по странице.
    """
    import fitz  # PyMuPDF — imported here to keep module importable without it

    doc = fitz.open(pdf_path)
    pages_text: List[str] = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: b[1])
        page_lines = [b[4].strip() for b in blocks if b[4].strip()]
        pages_text.append("\n".join(page_lines))
    return "\n".join(pages_text)


def detect_doc_type(text: str) -> str:
    """Return 'legal' if Russian regulatory keywords are found, else 'guide'.

    FR: Retourne 'legal' si ПОЛОЖЕНИЕ/ПРИКАЗ détecté, sinon 'guide'.
    RU: Возвращает 'legal' если найдены ПОЛОЖЕНИЕ/ПРИКАЗ, иначе 'guide'.
    """
    if LEGAL_DOC_PATTERN.search(text[:2000]):
        return "legal"
    return "guide"


def chunk_by_articles(text: str, source: str) -> List[Dict]:
    """Split text into numbered article chunks (for legal/regulatory docs).

    Refactored from extration.py:parser_le_document_ameliore.

    FR: Découpe par articles numérotés (ex. 3.1., 3.2.) pour docs officiels.
    RU: Разбивает по нумерованным статьям (напр. 3.1.) для официальных документов.
    """
    chunks: List[Dict] = []
    current_article: Optional[Dict] = None
    current_section: Optional[Dict] = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        sec_match = SECTION_PATTERN.match(line)
        if sec_match and not ARTICLE_PATTERN.match(line):
            current_section = {"num": sec_match.group(1), "titre": sec_match.group(2)}
            continue

        art_match = ARTICLE_PATTERN.match(line)
        if art_match:
            if current_article:
                current_article["content"] = re.sub(
                    r"\s+", " ", current_article["content"]
                ).strip()
                chunks.append(current_article)

            current_article = {
                "content": line,
                "source": source,
                "page": None,
                "type": "pdf",
                "article_num": art_match.group(1),
                "section_num": current_section["num"] if current_section else "",
                "section_titre": current_section["titre"] if current_section else "",
            }
        elif current_article:
            current_article["content"] += " " + line

    if current_article:
        current_article["content"] = re.sub(
            r"\s+", " ", current_article["content"]
        ).strip()
        chunks.append(current_article)

    return chunks


def chunk_by_tokens(
    text: str,
    source: str,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Dict]:
    """Sliding-window word chunking for prose documents (brochures, guides).

    FR: Fenêtre glissante sur les mots pour docs en prose (guides, brochures).
    RU: Скользящее окно по словам для прозаических документов (брошюры, справки).
    """
    # Split into paragraphs first to avoid splitting mid-sentence
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]

    # Flatten to a word list, keeping paragraph-break tracking simple
    all_words: List[str] = []
    for para in paragraphs:
        all_words.extend(para.split())

    chunks: List[Dict] = []
    start = 0
    total = len(all_words)

    while start < total:
        end = min(start + chunk_size, total)
        chunk_words = all_words[start:end]

        if len(chunk_words) < _MIN_CHUNK_WORDS and chunks:
            # Merge tiny tail into previous chunk
            chunks[-1]["content"] += " " + " ".join(chunk_words)
            break

        chunks.append(
            {
                "content": " ".join(chunk_words),
                "source": source,
                "page": None,
                "type": "pdf",
            }
        )

        # Advance by chunk_size - overlap (sliding window)
        start += chunk_size - overlap

    return chunks


def translate_chunks(
    chunks: List[Dict],
    src: str = "ru",
    tgt: str = "fr",
) -> List[Dict]:
    """Translate the 'content' field of each chunk using GoogleTranslator.

    FR: Traduit le champ 'content' de chaque chunk via GoogleTranslator.
    RU: Переводит поле 'content' каждого чанка через GoogleTranslator.
    """
    from deep_translator import GoogleTranslator  # imported here — optional dependency

    translator = GoogleTranslator(source=src, target=tgt)
    translated: List[Dict] = []

    for chunk in chunks:
        try:
            time.sleep(0.5)  # Avoid Google rate-limiting
            chunk["content"] = translator.translate(chunk["content"])
        except Exception as exc:  # noqa: BLE001
            chunk["content"] = f"[Erreur traduction: {exc}]"
        translated.append(chunk)

    return translated


def extract_and_chunk_pdf(
    pdf_path: str,
    translate: bool = True,
    chunk_size: int = 300,
    overlap: int = 50,
) -> List[Dict]:
    """Main entry point: extract → detect type → chunk → (optionally) translate.

    FR: Point d'entrée principal : extrait, détecte le type, découpe, traduit.
    RU: Основная точка входа: извлечение, определение типа, разбивка, перевод.

    Returns a list of chunk dicts ready for build_qdrant_points_pdf().
    """
    source = Path(pdf_path).name
    text = extract_text_from_pdf(pdf_path)
    doc_type = detect_doc_type(text)

    if doc_type == "legal":
        chunks = chunk_by_articles(text, source)
    else:
        chunks = chunk_by_tokens(text, source, chunk_size=chunk_size, overlap=overlap)

    if translate:
        chunks = translate_chunks(chunks)

    return chunks
