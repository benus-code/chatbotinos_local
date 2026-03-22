"""PDF chunking utilities for document indexation.

FR: Utilitaires de découpage de PDF pour l'indexation de documents.
RU: Утилиты разбивки PDF для индексации документов.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

LEGAL_KEYWORDS = ("положение", "приказ")
LEGAL_KEYWORD_WINDOW = 2000

ARTICLE_PATTERN = re.compile(r"^(\d+\.\d+)\.\s+(.+)$", re.MULTILINE)
SECTION_PATTERN = re.compile(r"^(\d+)\.\s+(.+)$", re.MULTILINE)


def detect_doc_type(text: str) -> str:
    """Detect whether a document is a legal act or a guide.

    FR: Détermine si le document est un acte légal ou un guide.
    RU: Определяет, является ли документ нормативным актом или руководством.
    """
    sample = text[:LEGAL_KEYWORD_WINDOW].lower()
    if any(kw in sample for kw in LEGAL_KEYWORDS):
        return "legal"
    return "guide"


def chunk_by_articles(text: str, source: str) -> List[Dict[str, Any]]:
    """Split legal document text into article-level chunks.

    FR: Découpe un texte juridique en blocs par article numéroté.
    RU: Разбивает юридический текст на блоки по пронумерованным статьям.
    """
    chunks: List[Dict[str, Any]] = []
    current_section: Optional[Dict[str, str]] = None

    lines = text.split("\n")
    current_article: Optional[Dict[str, Any]] = None

    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue

        sec_match = SECTION_PATTERN.match(line_stripped)
        art_match = ARTICLE_PATTERN.match(line_stripped)

        if art_match:
            if current_article is not None:
                chunks.append(current_article)
            current_article = {
                "content": line_stripped,
                "metadata": {
                    "article_num": art_match.group(1),
                    "article_title": art_match.group(2),
                    "section_num": current_section["num"] if current_section else "",
                    "section_title": current_section["title"] if current_section else "",
                },
                "source": source,
                "type": "pdf",
            }
        elif sec_match and not art_match:
            current_section = {"num": sec_match.group(1), "title": sec_match.group(2)}
        elif current_article is not None:
            current_article["content"] += " " + line_stripped

    if current_article is not None:
        chunks.append(current_article)

    return chunks


def chunk_by_tokens(
    text: str,
    source: str,
    doc_type: str,
    max_tokens: int = 200,
    overlap: int = 50,
) -> List[Dict[str, Any]]:
    """Split text into overlapping token-window chunks.

    FR: Découpe le texte en fenêtres de tokens avec chevauchement.
    RU: Разбивает текст на перекрывающиеся окна токенов.
    """
    words = text.split()
    if not words:
        return []

    chunks: List[Dict[str, Any]] = []
    start = 0

    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk_words = words[start:end]
        chunks.append(
            {
                "content": " ".join(chunk_words),
                "source": source,
                "type": doc_type,
            }
        )
        if end == len(words):
            break
        start += max_tokens - overlap

    # Merge tiny tail (< 20% of max_tokens) into the previous chunk
    if len(chunks) > 1 and len(chunks[-1]["content"].split()) < max_tokens // 5:
        tail = chunks.pop()
        chunks[-1]["content"] += " " + tail["content"]

    return chunks


def extract_and_chunk_pdf(
    path: str,
    translate: bool = True,
) -> List[Dict[str, Any]]:
    """Extract text from a PDF and split into chunks based on document type.

    FR: Extrait le texte d'un PDF et le découpe selon le type de document.
    RU: Извлекает текст из PDF и разбивает его в зависимости от типа документа.
    """
    import fitz  # type: ignore[import]

    doc = fitz.open(path)
    pages_text: List[str] = []
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        blocks = page.get_text("blocks")
        blocks.sort(key=lambda b: b[1])
        for block in blocks:
            text = block[4].strip()
            if text:
                pages_text.append(text)
    doc.close()

    full_text = "\n".join(pages_text)
    source = path.split("/")[-1]
    doc_type = detect_doc_type(full_text)

    if doc_type == "legal":
        chunks = chunk_by_articles(full_text, source)
    else:
        chunks = chunk_by_tokens(full_text, source, doc_type)

    if translate:
        # Translation placeholder — actual translation done externally.
        for chunk in chunks:
            chunk.setdefault("translated", False)

    return chunks
