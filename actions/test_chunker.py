"""Unit tests for LegislativeChunker.

Three test scenarios:
  1. Full legislative document — chapters + articles.
  2. Articles only — no chapter headers.
  3. Non-legislative prose — fallback sliding-window.

Run with:  python -m pytest actions/test_chunker.py -v
       or: python actions/test_chunker.py
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

# Allow running from repo root without installing the package.
sys.path.insert(0, str(Path(__file__).parent))

from chunker import LegislativeChunker


# ---------------------------------------------------------------------------
# Shared fixture texts
# ---------------------------------------------------------------------------

# 1. Full legislative document with chapters and articles (Гражданский кодекс excerpt)
LEGISLATIVE_FULL = """
ГРАЖДАНСКИЙ КОДЕКС РОССИЙСКОЙ ФЕДЕРАЦИИ

Глава 1. Гражданское законодательство

Статья 1. Основные начала гражданского законодательства
1. Гражданское законодательство основывается на признании равенства участников
регулируемых им отношений, неприкосновенности собственности, свободы договора,
недопустимости произвольного вмешательства кого-либо в частные дела.
2. Граждане и юридические лица приобретают и осуществляют свои гражданские права
своей волей и в своём интересе.

Статья 2. Отношения, регулируемые гражданским законодательством
1. Гражданское законодательство определяет правовое положение участников гражданского
оборота, основания возникновения и порядок осуществления права собственности.
2. Участниками регулируемых гражданским законодательством отношений являются граждане
и юридические лица.

Глава 2. Возникновение гражданских прав и обязанностей

Статья 8. Основания возникновения гражданских прав и обязанностей
1. Гражданские права и обязанности возникают из оснований, предусмотренных законом.
2. Гражданские права и обязанности возникают из договоров и иных сделок,
предусмотренных законом.
3. Гражданские права и обязанности возникают из актов государственных органов
и органов местного самоуправления.

Статья 9. Осуществление гражданских прав
1. Граждане и юридические лица по своему усмотрению осуществляют принадлежащие им
гражданские права.
2. Отказ граждан и юридических лиц от осуществления принадлежащих им прав не влечёт
прекращения этих прав.
""".strip()

# 2. Articles only — no Глава headers (e.g. a standalone decree)
LEGISLATIVE_NO_CHAPTERS = """
ФЕДЕРАЛЬНЫЙ ЗАКОН О ПЕРСОНАЛЬНЫХ ДАННЫХ

Статья 1. Сфера действия настоящего Федерального закона
1. Настоящий Федеральный закон регулирует отношения, связанные с обработкой
персональных данных, осуществляемой федеральными органами государственной власти.
2. Действие настоящего Федерального закона не распространяется на отношения,
возникающие при обработке персональных данных физическими лицами исключительно
для личных и семейных нужд.

Статья 2. Цель настоящего Федерального закона
Целью настоящего Федерального закона является обеспечение защиты прав и свобод
человека и гражданина при обработке его персональных данных, в том числе защиты
прав на неприкосновенность частной жизни, личную и семейную тайну.

СТАТЬЯ 3. Основные понятия, используемые в настоящем Федеральном законе
1. В целях настоящего Федерального закона используются следующие основные понятия:
персональные данные — любая информация, относящаяся к прямо или косвенно
определённому или определяемому физическому лицу.
""".strip()

# 3. Non-legislative prose — triggers fallback chunking
PROSE_TEXT = " ".join([
    "Студенческое общежитие предоставляет жилые помещения иногородним студентам.",
    "Заселение производится на основании направления, выданного деканатом факультета.",
    "Каждый студент обязан соблюдать правила внутреннего распорядка общежития.",
    "Запрещается курение в комнатах и местах общего пользования.",
    "Студенты несут ответственность за сохранность имущества общежития.",
    "При выезде из общежития необходимо сдать комнату коменданту.",
    "Оплата за проживание производится ежемесячно до пятого числа текущего месяца.",
    "Студенты, нарушающие правила проживания, могут быть выселены из общежития.",
] * 40)   # ~320 words → triggers at least one fallback chunk


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

class TestLegislativeChunker(unittest.TestCase):

    def setUp(self) -> None:
        self.chunker = LegislativeChunker()

    # ------------------------------------------------------------------
    # Scenario 1 — Full legislative document
    # ------------------------------------------------------------------

    def test_full_doc_returns_chunks(self) -> None:
        """At least one chunk is produced for the full document."""
        chunks = self.chunker.chunk(LEGISLATIVE_FULL)
        self.assertGreater(len(chunks), 0)

    def test_full_doc_chunk_count(self) -> None:
        """5 chunks: 1 preamble + 4 articles (2 chapters × 2 articles each)."""
        chunks = self.chunker.chunk(LEGISLATIVE_FULL)
        self.assertEqual(len(chunks), 5)

    def test_full_doc_chapter_propagation(self) -> None:
        """Each chunk carries the chapter it belongs to."""
        chunks = self.chunker.chunk(LEGISLATIVE_FULL)
        chapters = {c["metadata"]["chapter"] for c in chunks}
        # Both chapters must appear
        self.assertTrue(
            any("Глава 1" in ch for ch in chapters),
            f"Глава 1 not found in chapters: {chapters}",
        )
        self.assertTrue(
            any("Глава 2" in ch for ch in chapters),
            f"Глава 2 not found in chapters: {chapters}",
        )

    def test_full_doc_article_metadata(self) -> None:
        """Article metadata matches the header text."""
        chunks = self.chunker.chunk(LEGISLATIVE_FULL)
        articles = [c["metadata"]["article"] for c in chunks]
        self.assertTrue(
            any("Статья 1" in a for a in articles),
            f"Статья 1 not found: {articles}",
        )
        self.assertTrue(
            any("Статья 8" in a for a in articles),
            f"Статья 8 not found: {articles}",
        )

    def test_full_doc_type_legislative(self) -> None:
        """All chunks produced from a legislative document have type='legislative'."""
        chunks = self.chunker.chunk(LEGISLATIVE_FULL)
        for chunk in chunks:
            self.assertEqual(chunk["metadata"]["type"], "legislative")

    def test_full_doc_text_contains_content(self) -> None:
        """Each chunk text is non-empty and contains substantive content."""
        chunks = self.chunker.chunk(LEGISLATIVE_FULL)
        for chunk in chunks:
            self.assertIsInstance(chunk["text"], str)
            self.assertGreater(len(chunk["text"].strip()), 10)

    def test_full_doc_chunk_schema(self) -> None:
        """Every chunk has exactly the expected keys and metadata sub-keys."""
        chunks = self.chunker.chunk(LEGISLATIVE_FULL)
        for chunk in chunks:
            self.assertIn("text", chunk)
            self.assertIn("metadata", chunk)
            meta = chunk["metadata"]
            self.assertIn("article", meta)
            self.assertIn("chapter", meta)
            self.assertIn("type", meta)

    # ------------------------------------------------------------------
    # Scenario 2 — Articles only, no chapters
    # ------------------------------------------------------------------

    def test_no_chapters_chunk_count(self) -> None:
        """4 chunks: 1 preamble + 3 articles (no chapter headers)."""
        chunks = self.chunker.chunk(LEGISLATIVE_NO_CHAPTERS)
        self.assertEqual(len(chunks), 4)

    def test_no_chapters_empty_chapter_field(self) -> None:
        """Chapter metadata is an empty string when no Глава headers are present."""
        chunks = self.chunker.chunk(LEGISLATIVE_NO_CHAPTERS)
        for chunk in chunks:
            self.assertEqual(chunk["metadata"]["chapter"], "")

    def test_no_chapters_article_titles(self) -> None:
        """Article titles are captured correctly including uppercase СТАТЬЯ."""
        chunks = self.chunker.chunk(LEGISLATIVE_NO_CHAPTERS)
        articles = [c["metadata"]["article"] for c in chunks]
        self.assertTrue(any("Статья 1" in a for a in articles))
        self.assertTrue(any("Статья 2" in a for a in articles))
        # СТАТЬЯ (uppercase) must also be captured
        self.assertTrue(
            any("СТАТЬЯ 3" in a or "Статья 3" in a for a in articles),
            f"Article 3 not captured: {articles}",
        )

    def test_no_chapters_type_legislative(self) -> None:
        chunks = self.chunker.chunk(LEGISLATIVE_NO_CHAPTERS)
        for chunk in chunks:
            self.assertEqual(chunk["metadata"]["type"], "legislative")

    def test_no_chapters_text_not_empty(self) -> None:
        chunks = self.chunker.chunk(LEGISLATIVE_NO_CHAPTERS)
        for chunk in chunks:
            self.assertGreater(len(chunk["text"].strip()), 0)

    # ------------------------------------------------------------------
    # Scenario 3 — Non-legislative prose → fallback
    # ------------------------------------------------------------------

    def test_fallback_produces_chunks(self) -> None:
        """Non-legislative text is split into at least one chunk."""
        chunks = self.chunker.chunk(PROSE_TEXT)
        self.assertGreater(len(chunks), 0)

    def test_fallback_type_paragraph(self) -> None:
        """Fallback chunks have type='paragraph'."""
        chunks = self.chunker.chunk(PROSE_TEXT)
        for chunk in chunks:
            self.assertEqual(chunk["metadata"]["type"], "paragraph")

    def test_fallback_empty_article_and_chapter(self) -> None:
        """Fallback chunks have empty article and chapter fields."""
        chunks = self.chunker.chunk(PROSE_TEXT)
        for chunk in chunks:
            self.assertEqual(chunk["metadata"]["article"], "")
            self.assertEqual(chunk["metadata"]["chapter"], "")

    def test_fallback_window_size(self) -> None:
        """Each fallback chunk contains at most FALLBACK_WINDOW words."""
        chunks = self.chunker.chunk(PROSE_TEXT)
        for chunk in chunks:
            word_count = len(chunk["text"].split())
            self.assertLessEqual(
                word_count,
                LegislativeChunker.FALLBACK_WINDOW,
                f"Chunk exceeds window: {word_count} words",
            )

    def test_fallback_overlap(self) -> None:
        """Consecutive fallback chunks share an overlapping suffix/prefix."""
        chunks = self.chunker.chunk(PROSE_TEXT)
        if len(chunks) < 2:
            self.skipTest("Not enough chunks to test overlap")

        window = LegislativeChunker.FALLBACK_WINDOW
        overlap = LegislativeChunker.FALLBACK_OVERLAP

        words_0 = chunks[0]["text"].split()
        words_1 = chunks[1]["text"].split()

        # Last `overlap` words of chunk 0 should equal first `overlap` words of chunk 1
        self.assertEqual(
            words_0[window - overlap:],
            words_1[:overlap],
        )

    def test_fallback_coverage(self) -> None:
        """All words in the original text appear in at least one chunk."""
        chunks = self.chunker.chunk(PROSE_TEXT)
        all_chunk_text = " ".join(c["text"] for c in chunks)
        original_words = set(PROSE_TEXT.split())
        chunk_words = set(all_chunk_text.split())
        missing = original_words - chunk_words
        self.assertEqual(missing, set(), f"Missing words from chunks: {missing}")

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_empty_string(self) -> None:
        """Empty input returns an empty list."""
        self.assertEqual(self.chunker.chunk(""), [])

    def test_whitespace_only(self) -> None:
        """Whitespace-only input returns an empty list."""
        self.assertEqual(self.chunker.chunk("   \n\t  "), [])

    def test_single_article_no_body(self) -> None:
        """An article header with no body still produces one chunk."""
        text = "Статья 1. Краткая норма"
        chunks = self.chunker.chunk(text)
        self.assertEqual(len(chunks), 1)
        self.assertIn("Статья 1", chunks[0]["metadata"]["article"])


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
