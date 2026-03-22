"""Unit tests for pdf_chunker.py.

Tests run without a real PDF file, GPU, or network access.
fitz (PyMuPDF) and deep_translator are mocked where needed.

FR: Tests unitaires pour pdf_chunker.py.
RU: Модульные тесты для pdf_chunker.py.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# detect_doc_type
# ---------------------------------------------------------------------------

class TestDetectDocType:
    def setup_method(self):
        from pdf_chunker import detect_doc_type
        self.detect = detect_doc_type

    def test_legal_for_polozhenie(self):
        text = "ПОЛОЖЕНИЕ о студенческом общежитии ТУСУР"
        assert self.detect(text) == "legal"

    def test_legal_for_prikaz(self):
        text = "ПРИКАЗ ректора № 123"
        assert self.detect(text) == "legal"

    def test_guide_for_prose(self):
        text = "Bienvenue à l'université. Ce guide vous aidera à vous intégrer."
        assert self.detect(text) == "guide"

    def test_guide_when_keywords_beyond_2000_chars(self):
        # Keyword appears only after the 2000-char search window → 'guide'
        text = "x" * 2001 + " ПОЛОЖЕНИЕ"
        assert self.detect(text) == "guide"


# ---------------------------------------------------------------------------
# chunk_by_articles
# ---------------------------------------------------------------------------

class TestChunkByArticles:
    def setup_method(self):
        from pdf_chunker import chunk_by_articles
        self.chunk = chunk_by_articles

    def test_extracts_numbered_articles(self):
        text = (
            "1. Dispositions générales\n"
            "1.1. Le présent règlement définit les droits des étudiants.\n"
            "1.2. Les étudiants sont tenus de respecter les règles.\n"
        )
        chunks = self.chunk(text, "reglement.pdf")
        assert len(chunks) == 2
        assert chunks[0]["article_num"] == "1.1"
        assert chunks[1]["article_num"] == "1.2"

    def test_section_metadata_attached(self):
        text = (
            "2. Droits et obligations\n"
            "2.1. Les étudiants ont le droit de résider en cité.\n"
        )
        chunks = self.chunk(text, "reglement.pdf")
        assert chunks[0]["section_num"] == "2"
        assert "Droits" in chunks[0]["section_titre"]

    def test_source_field_set(self):
        text = "3.1. Article de test.\n"
        chunks = self.chunk(text, "mon_doc.pdf")
        assert chunks[0]["source"] == "mon_doc.pdf"

    def test_type_is_pdf(self):
        text = "4.1. Contenu quelconque.\n"
        chunks = self.chunk(text, "doc.pdf")
        assert chunks[0]["type"] == "pdf"

    def test_no_articles_returns_empty(self):
        text = "Texte sans aucun numéro d'article."
        chunks = self.chunk(text, "doc.pdf")
        assert chunks == []


# ---------------------------------------------------------------------------
# chunk_by_tokens
# ---------------------------------------------------------------------------

class TestChunkByTokens:
    def setup_method(self):
        from pdf_chunker import chunk_by_tokens
        self.chunk = chunk_by_tokens

    def _make_long_text(self, word_count: int) -> str:
        return " ".join([f"mot{i}" for i in range(word_count)])

    def test_single_chunk_for_short_text(self):
        text = self._make_long_text(100)
        chunks = self.chunk(text, "guide.pdf", chunk_size=300, overlap=50)
        assert len(chunks) == 1

    def test_multiple_chunks_for_long_text(self):
        text = self._make_long_text(700)
        chunks = self.chunk(text, "guide.pdf", chunk_size=300, overlap=50)
        # 700 words / (300 - 50 step) ≈ 3 chunks
        assert len(chunks) >= 2

    def test_overlap_present_between_chunks(self):
        text = self._make_long_text(400)
        chunks = self.chunk(text, "guide.pdf", chunk_size=200, overlap=40)
        if len(chunks) >= 2:
            words_1 = chunks[0]["content"].split()
            words_2 = chunks[1]["content"].split()
            # The last 40 words of chunk 0 should appear at the start of chunk 1
            tail = words_1[-40:]
            head = words_2[:40]
            assert tail == head

    def test_source_and_type_fields(self):
        text = self._make_long_text(100)
        chunk = self.chunk(text, "guide.pdf")[0]
        assert chunk["source"] == "guide.pdf"
        assert chunk["type"] == "pdf"

    def test_tiny_tail_merged_into_previous(self):
        # 310 words: first chunk 300, remaining 10 → merged
        text = self._make_long_text(310)
        chunks = self.chunk(text, "guide.pdf", chunk_size=300, overlap=0)
        assert len(chunks) == 1


# ---------------------------------------------------------------------------
# extract_and_chunk_pdf (integration — fitz + translator mocked)
# ---------------------------------------------------------------------------

class TestExtractAndChunkPdf:
    def test_legal_doc_uses_article_chunking(self, tmp_path):
        fake_pdf = tmp_path / "polozhenie.pdf"
        fake_pdf.write_bytes(b"fake pdf content")

        legal_text = (
            "ПОЛОЖЕНИЕ о студенческом общежитии\n"
            "1. Dispositions générales\n"
            "1.1. Les étudiants résident à la cité.\n"
            "1.2. Les règles sont obligatoires.\n"
        )

        with (
            patch("pdf_chunker.extract_text_from_pdf", return_value=legal_text),
            patch("pdf_chunker.translate_chunks", side_effect=lambda c, **kw: c),
        ):
            from pdf_chunker import extract_and_chunk_pdf
            chunks = extract_and_chunk_pdf(str(fake_pdf), translate=True)

        assert len(chunks) == 2
        assert chunks[0]["article_num"] == "1.1"

    def test_guide_doc_uses_token_chunking(self, tmp_path):
        fake_pdf = tmp_path / "guide_etudiant.pdf"
        fake_pdf.write_bytes(b"fake pdf content")

        prose_text = " ".join([f"mot{i}" for i in range(400)])

        with (
            patch("pdf_chunker.extract_text_from_pdf", return_value=prose_text),
            patch("pdf_chunker.translate_chunks", side_effect=lambda c, **kw: c),
        ):
            from pdf_chunker import extract_and_chunk_pdf
            chunks = extract_and_chunk_pdf(str(fake_pdf), translate=True)

        assert len(chunks) >= 1
        assert chunks[0]["type"] == "pdf"
        assert "article_num" not in chunks[0]

    def test_no_translate_skips_translation(self, tmp_path):
        fake_pdf = tmp_path / "doc.pdf"
        fake_pdf.write_bytes(b"fake pdf content")
        prose_text = " ".join([f"word{i}" for i in range(100)])

        with (
            patch("pdf_chunker.extract_text_from_pdf", return_value=prose_text),
            patch("pdf_chunker.translate_chunks") as mock_translate,
        ):
            from pdf_chunker import extract_and_chunk_pdf
            extract_and_chunk_pdf(str(fake_pdf), translate=False)

        mock_translate.assert_not_called()
