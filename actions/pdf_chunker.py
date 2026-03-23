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


_MAX_ARTICLE_WORDS = 200  # Articles > 200 mots sont découpés en sous-chunks


def _split_article_into_subchunks(article: Dict) -> List[Dict]:
    """Split an oversized article into sentence-based sub-chunks.

    FR: Découpe un article trop long en sous-chunks basés sur les phrases.
    RU: Разбивает слишком длинную статью на подчанки по предложениям.
    """
    words = article["content"].split()
    if len(words) <= _MAX_ARTICLE_WORDS:
        return [article]

    # Split on sentence boundaries (". ", "; ", ": ")
    sentences = re.split(r"(?<=[.;:])\s+", article["content"])
    subchunks: List[Dict] = []
    current_words: List[str] = []
    sub_idx = 1

    for sentence in sentences:
        sentence_words = sentence.split()
        if current_words and len(current_words) + len(sentence_words) > _MAX_ARTICLE_WORDS:
            subchunks.append({
                **article,
                "content": " ".join(current_words),
                "article_num": f"{article['article_num']}.{sub_idx}",
            })
            sub_idx += 1
            current_words = sentence_words
        else:
            current_words.extend(sentence_words)

    if current_words:
        subchunks.append({
            **article,
            "content": " ".join(current_words),
            "article_num": f"{article['article_num']}.{sub_idx}",
        })

    return subchunks


def chunk_by_articles(text: str, source: str) -> List[Dict]:
    """Split text into numbered article chunks (for legal/regulatory docs).

    Articles longer than _MAX_ARTICLE_WORDS are further split into
    sentence-based sub-chunks to improve translation quality and
    retrieval precision.

    FR: Découpe par articles numérotés (ex. 3.1., 3.2.) pour docs officiels.
        Les articles > 200 mots sont découpés en sous-chunks par phrase.
    RU: Разбивает по нумерованным статьям для официальных документов.
        Статьи > 200 слов дополнительно разбиваются по предложениям.
    """
    raw_chunks: List[Dict] = []
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
                raw_chunks.append(current_article)

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
        raw_chunks.append(current_article)

    # Split oversized articles into sentence-based sub-chunks
    chunks: List[Dict] = []
    for article in raw_chunks:
        chunks.extend(_split_article_into_subchunks(article))

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
