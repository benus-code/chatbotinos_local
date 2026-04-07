"""PDF text extraction and chunking strategies for Qdrant indexation.

Two strategies are supported:
- 'legal'  : article-based chunking for Russian regulatory docs (ПОЛОЖЕНИЕ, ПРИКАЗ).
- 'guide'  : token-based sliding-window for prose documents (student brochures).

FR: Extraction et découpage de PDFs selon le type de document.
RU: Извлечение текста и разбивка PDF в зависимости от типа документа.
"""

from __future__ import annotations

import html
import re
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


_MAX_SECTION_WORDS = 500  # Section > 500 mots → découpe en deux demi-sections


def chunk_by_sections(text: str, source: str) -> List[Dict]:
    """Regroupe tous les sous-articles d'un même article principal en un seul chunk.

    FR: Au lieu de 1 chunk par sous-article (1.1, 1.2...), produit 1 chunk par
    article principal (tout l'Article 1, tout l'Article 2, etc.).
    Chaque chunk est ainsi auto-suffisant et contient le contexte complet.
    RU: Вместо одного чанка на подстатью (1.1, 1.2...) создаёт один чанк
    на главную статью (вся Статья 1, вся Статья 2 и т.д.).
    Каждый чанк самодостаточен и содержит полный контекст.
    """
    sections: Dict[str, Dict] = {}  # section_num → {"titre": ..., "content": ..., "order": ...}
    current_section_num: Optional[str] = None
    order = 0

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue

        # En-tête de section principale : "2. Права и обязанности..."
        sec_match = SECTION_PATTERN.match(line)
        if sec_match and not ARTICLE_PATTERN.match(line):
            current_section_num = sec_match.group(1)
            if current_section_num not in sections:
                sections[current_section_num] = {
                    "titre": sec_match.group(2),
                    "content": "",
                    "order": order,
                }
                order += 1
            continue

        # Sous-article : "2.1. ..." → rattaché à la section parente
        art_match = ARTICLE_PATTERN.match(line)
        if art_match:
            parent = art_match.group(1).split(".")[0]
            if parent not in sections:
                sections[parent] = {"titre": "", "content": "", "order": order}
                order += 1
            sections[parent]["content"] += " " + line
            current_section_num = parent
        elif current_section_num and current_section_num in sections:
            sections[current_section_num]["content"] += " " + line

    # Construit les chunks finaux, un par section principale
    # Формирует финальные чанки — по одному на каждую главную секцию
    chunks: List[Dict] = []
    for sec_num, sec in sorted(sections.items(), key=lambda x: x[1]["order"]):
        content = re.sub(r"\s+", " ", sec["content"]).strip()
        if not content or len(content.split()) < 10:
            continue

        base_chunk = {
            "source": source,
            "page": None,
            "type": "pdf",
            "section_num": sec_num,
            "section_titre": sec["titre"],
        }

        words = content.split()
        if len(words) <= _MAX_SECTION_WORDS:
            chunks.append({**base_chunk, "content": content})
        else:
            # Coupe en deux moitiés pour les sections très longues
            # Разрезаем пополам для очень длинных секций
            mid = len(words) // 2
            chunks.append({**base_chunk, "content": " ".join(words[:mid]), "section_num": f"{sec_num}a"})
            chunks.append({**base_chunk, "content": " ".join(words[mid:]), "section_num": f"{sec_num}b"})

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


_TRANSLATOR = None  # None = pas chargé ; False = échec ; tuple = (tokenizer, model)

def _get_marian_translator():
    """Charge Helsinki-NLP/opus-mt-ru-fr une seule fois (lazy loading)."""
    global _TRANSLATOR
    if _TRANSLATOR is False:
        raise RuntimeError("Le modèle de traduction n'a pas pu être chargé.")
    if _TRANSLATOR is None:
        from transformers import MarianMTModel, MarianTokenizer
        model_name = "Helsinki-NLP/opus-mt-ru-fr"
        print(f"Chargement du modele de traduction {model_name}...")
        try:
            tokenizer = MarianTokenizer.from_pretrained(model_name)
            model = MarianMTModel.from_pretrained(model_name)
            _TRANSLATOR = (tokenizer, model)
            print("Modele de traduction pret.")
        except Exception as exc:
            _TRANSLATOR = False
            raise RuntimeError(f"Impossible de charger {model_name}: {exc}") from exc
    return _TRANSLATOR


def translate_chunks(
    chunks: List[Dict],
    src: str = "ru",
    tgt: str = "fr",
) -> List[Dict]:
    """Translate the 'content' field of each chunk using Helsinki-NLP/opus-mt-ru-fr (offline).

    FR: Traduit le champ 'content' de chaque chunk via MarianMT (offline).
    RU: Переводит поле 'content' каждого чанка через MarianMT (офлайн).
    """
    for chunk in chunks:
        try:
            tokenizer, model = _get_marian_translator()
            inputs = tokenizer(
                chunk["content"],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            )
            translated = model.generate(
                **inputs,
                max_new_tokens=512,
                no_repeat_ngram_size=3,
                early_stopping=True,
                num_beams=4,
            )
            raw = tokenizer.decode(translated[0], skip_special_tokens=True)
            raw = html.unescape(raw)
            # Fix spaces around apostrophes: "d ' hébergement" → "d'hébergement"
            raw = re.sub(r"\s+'\s+", "'", raw)
            # Remove leftover HTML entity fragments (e.g. "&atation", "apos;")
            raw = re.sub(r"&[a-zA-Z]+;?", "", raw)
            raw = re.sub(r"\b[a-z]+;", "", raw)
            chunk["content"] = raw.strip()
        except Exception as exc:  # noqa: BLE001
            print(f"[ERREUR TRADUCTION] {exc}")
            chunk["content"] = f"[Erreur traduction: {exc}]"

    return chunks


def extract_and_chunk_pdf(
    pdf_path: str,
    translate: bool = False,  # Approche C : on stocke le russe natif, traduction à l'affichage. / Подход C: храним русский текст, переводим при отображении.
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
        # chunk_by_sections : 1 chunk par article principal (tout l'art. 2 ensemble).
        # chunk_by_sections: 1 chunk per main article (all of art. 2 together).
        chunks = chunk_by_sections(text, source)
    else:
        chunks = chunk_by_tokens(text, source, chunk_size=chunk_size, overlap=overlap)

    if translate:
        chunks = translate_chunks(chunks)

    return chunks


if __name__ == "__main__":
    import logging
    import os
    import sys

    from qdrant_client import QdrantClient
    from sentence_transformers import SentenceTransformer

    from indexation import (
        build_qdrant_points_pdf,
        replace_pdf_source_points,
        collection_name,
    )

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    LOGGER = logging.getLogger(__name__)

    # Fichier PDF à indexer — situé dans le même répertoire que ce script.
    # PDF-файл для индексации — находится в той же папке, что и этот скрипт.
    _here = Path(__file__).parent
    pdf_filename = "Polozhenie_o_studencheskom_obschezhitii.pdf"
    pdf_path = _here / pdf_filename

    if not pdf_path.exists():
        LOGGER.error("PDF non trouvé : %s", pdf_path)
        sys.exit(1)

    LOGGER.info("Extraction et découpage du PDF : %s", pdf_filename)
    chunks = extract_and_chunk_pdf(str(pdf_path), translate=False)
    LOGGER.info("%d chunks extraits.", len(chunks))

    qdrant_host = os.getenv("QDRANT_HOST", "qdrant")
    qdrant_client = QdrantClient(host=qdrant_host, port=6333, timeout=60)

    # Modèle multilingue partagé avec indexation.py.
    # Многоязычная модель, общая с indexation.py.
    embedding_model = SentenceTransformer("intfloat/multilingual-e5-large")

    points = build_qdrant_points_pdf(chunks, embedding_model)
    LOGGER.info("%d points Qdrant construits.", len(points))

    replace_pdf_source_points(
        qdrant_client=qdrant_client,
        collection=collection_name,
        source_name=pdf_filename,
        new_points=points,
    )
    LOGGER.info("PDF '%s' indexé dans la collection '%s'.", pdf_filename, collection_name)
