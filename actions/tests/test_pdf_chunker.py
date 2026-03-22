"""Tests for PDF chunking utilities.

FR: Tests unitaires pour les utilitaires de découpage de PDF.
RU: Юнит-тесты для утилит разбивки PDF.
"""

import pytest
from unittest.mock import patch, MagicMock


class TestDetectDocType:
    def detect(self, text):
        from pdf_chunker import detect_doc_type
        return detect_doc_type(text)

    def test_legal_for_polozhenie(self):
        assert self.detect("положение об общежитии") == "legal"

    def test_legal_for_prikaz(self):
        assert self.detect("приказ ректора") == "legal"

    def test_guide_for_prose(self):
        assert self.detect("This is a student guide about university life.") == "guide"

    def test_guide_when_keywords_beyond_2000_chars(self):
        padding = "x" * 2001
        text = padding + " положение"
        assert self.detect(text) == "guide"


class TestChunkByArticles:
    LEGAL_TEXT = (
        "1. Общие положения\n"
        "1.1. Настоящее положение регулирует порядок.\n"
        "1.2. Действие распространяется на всех студентов.\n"
        "2. Права и обязанности\n"
        "2.1. Студенты имеют право на проживание.\n"
    )

    def chunks(self, text=None, source="test.pdf"):
        from pdf_chunker import chunk_by_articles
        return chunk_by_articles(text or self.LEGAL_TEXT, source)

    def test_extracts_numbered_articles(self):
        result = self.chunks()
        assert len(result) == 3  # 1.1, 1.2, 2.1

    def test_section_metadata_attached(self):
        result = self.chunks()
        first = result[0]
        assert first["metadata"]["section_num"] == "1"

    def test_source_field_set(self):
        result = self.chunks(source="doc.pdf")
        assert all(c["source"] == "doc.pdf" for c in result)

    def test_type_is_pdf(self):
        result = self.chunks()
        assert all(c["type"] == "pdf" for c in result)

    def test_no_articles_returns_empty(self):
        result = self.chunks(text="Just some prose without article numbers.")
        assert result == []


class TestChunkByTokens:
    SHORT_TEXT = "Hello world this is a short text."
    LONG_TEXT = " ".join([f"word{i}" for i in range(500)])

    def chunks(self, text, source="doc.pdf", doc_type="guide", max_tokens=200, overlap=50):
        from pdf_chunker import chunk_by_tokens
        return chunk_by_tokens(text, source, doc_type, max_tokens, overlap)

    def test_single_chunk_for_short_text(self):
        result = self.chunks(self.SHORT_TEXT)
        assert len(result) == 1

    def test_multiple_chunks_for_long_text(self):
        result = self.chunks(self.LONG_TEXT)
        assert len(result) > 1

    def test_overlap_present_between_chunks(self):
        result = self.chunks(self.LONG_TEXT, max_tokens=100, overlap=20)
        assert len(result) >= 2
        words_first = result[0]["content"].split()
        words_second = result[1]["content"].split()
        # The tail of first chunk overlaps with start of second
        overlap_words = set(words_first[-20:]) & set(words_second[:20])
        assert len(overlap_words) > 0

    def test_source_and_type_fields(self):
        result = self.chunks(self.SHORT_TEXT, source="my.pdf", doc_type="guide")
        assert result[0]["source"] == "my.pdf"
        assert result[0]["type"] == "guide"

    def test_tiny_tail_merged_into_previous(self):
        # Create text where last chunk would be tiny (< max_tokens // 5 words)
        # 210 words: chunks of 200 → first chunk 200 words, tail 10 words (< 200//5=40)
        text = " ".join([f"w{i}" for i in range(210)])
        result = self.chunks(text, max_tokens=200, overlap=0)
        assert len(result) == 1


class TestExtractAndChunkPdf:
    def _make_mock_doc(self, text):
        mock_page = MagicMock()
        mock_page.get_text.return_value = [
            (0, 0, 100, 20, text, 0, 0)
        ]
        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.load_page.return_value = mock_page
        return mock_doc

    def test_legal_doc_uses_article_chunking(self):
        legal_text = (
            "положение об общежитии\n"
            "1. Общие положения\n"
            "1.1. Студенты проживают в общежитии.\n"
            "1.2. Правила обязательны к исполнению.\n"
        )
        mock_doc = self._make_mock_doc(legal_text)
        with patch("fitz.open", return_value=mock_doc):
            from pdf_chunker import extract_and_chunk_pdf
            chunks = extract_and_chunk_pdf("test.pdf", translate=False)
        assert len(chunks) > 0
        assert all(c["type"] == "pdf" for c in chunks)

    def test_guide_doc_uses_token_chunking(self):
        guide_text = "This is a helpful student guide. " * 50
        mock_doc = self._make_mock_doc(guide_text)
        with patch("fitz.open", return_value=mock_doc):
            from pdf_chunker import extract_and_chunk_pdf
            chunks = extract_and_chunk_pdf("guide.pdf", translate=False)
        assert len(chunks) > 0
        assert all(c["type"] == "guide" for c in chunks)

    def test_no_translate_skips_translation(self):
        text = "1.1. Some legal article content.\n"
        mock_doc = self._make_mock_doc(text)
        with patch("fitz.open", return_value=mock_doc):
            from pdf_chunker import extract_and_chunk_pdf
            chunks = extract_and_chunk_pdf("doc.pdf", translate=False)
        # When translate=False, no 'translated' key is added
        assert all("translated" not in c for c in chunks)
