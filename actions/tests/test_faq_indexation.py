"""Unit tests for indexation.py (FAQ pipeline).

All tests run without a real Qdrant instance or embedding model.

FR: Tests unitaires pour indexation.py — pipeline FAQ.
RU: Модульные тесты для indexation.py — конвейер FAQ.
"""

from __future__ import annotations

import pytest

# indexation.py executes module-level code (model load + Qdrant connect) when
# imported directly.  We patch those before the import to keep tests fast.
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# load_and_split_faq
# ---------------------------------------------------------------------------

class TestLoadAndSplitFaq:
    def test_returns_blocks_for_valid_file(self, sample_faq_file):
        with patch("indexation.model", MagicMock()), patch("indexation.client", MagicMock()):
            from indexation import load_and_split_faq
            blocks = load_and_split_faq(sample_faq_file)
        # The sample FAQ has 3 Q- blocks
        assert len(blocks) == 3

    def test_each_block_starts_with_q(self, sample_faq_file):
        with patch("indexation.model", MagicMock()), patch("indexation.client", MagicMock()):
            from indexation import load_and_split_faq
            blocks = load_and_split_faq(sample_faq_file)
        assert all(b.startswith("Q-") for b in blocks)

    def test_raises_for_missing_file(self, tmp_path):
        with patch("indexation.model", MagicMock()), patch("indexation.client", MagicMock()):
            from indexation import load_and_split_faq
            with pytest.raises(FileNotFoundError):
                load_and_split_faq(str(tmp_path / "does_not_exist.txt"))


# ---------------------------------------------------------------------------
# parse_faq_block
# ---------------------------------------------------------------------------

class TestParseFaqBlock:
    def setup_method(self):
        with patch("indexation.model", MagicMock()), patch("indexation.client", MagicMock()):
            from indexation import parse_faq_block
            self.parse = parse_faq_block

    def test_extracts_question_and_answer(self):
        block = 'Q-"Comment obtenir l\'invitation ?"\nR- Payer 10 % des frais.'
        result = self.parse(block)
        assert result["question"] == "Comment obtenir l'invitation ?"
        assert "Payer 10 %" in result["answer"]

    def test_empty_answer(self):
        block = 'Q-"Comment prolonger le visa ?"\nR-'
        result = self.parse(block)
        assert result["question"] == "Comment prolonger le visa ?"
        assert result["answer"] == ""

    def test_missing_question_marker(self):
        block = "Juste un texte sans Q-"
        result = self.parse(block)
        assert result["question"] == ""


# ---------------------------------------------------------------------------
# build_qdrant_points
# ---------------------------------------------------------------------------

class TestBuildQdrantPoints:
    def setup_method(self):
        with patch("indexation.model", MagicMock()), patch("indexation.client", MagicMock()):
            from indexation import build_qdrant_points
            self.build = build_qdrant_points

    def test_returns_correct_count(self, mock_model):
        items = [
            {"question": "Q1 ?", "answer": "R1."},
            {"question": "Q2 ?", "answer": "R2."},
        ]
        points = self.build(items, mock_model, "FAQ.txt")
        assert len(points) == 2

    def test_payload_type_is_faq(self, mock_model):
        items = [{"question": "Q ?", "answer": "R."}]
        point = self.build(items, mock_model, "FAQ.txt")[0]
        assert point.payload["type"] == "faq"

    def test_payload_source_file(self, mock_model):
        items = [{"question": "Q ?", "answer": "R."}]
        point = self.build(items, mock_model, "FAQ.txt")[0]
        assert point.payload["source_file"] == "FAQ.txt"

    def test_vector_has_correct_dimension(self, mock_model):
        items = [{"question": "Q ?", "answer": "R."}]
        point = self.build(items, mock_model, "FAQ.txt")[0]
        assert len(point.vector) == 1024


# ---------------------------------------------------------------------------
# search_faq
# ---------------------------------------------------------------------------

class TestSearchFaq:
    def setup_method(self):
        with patch("indexation.model", MagicMock()), patch("indexation.client", MagicMock()):
            from indexation import search_faq
            self.search = search_faq

    def test_returns_points_for_valid_query(self, mock_qdrant, mock_model):
        results = self.search(mock_qdrant, "FAQ_Multilingue", mock_model, "visa étudiant")
        assert len(results) == 2
        assert results[0].score == 0.72

    def test_empty_query_returns_empty_list(self, mock_qdrant, mock_model):
        results = self.search(mock_qdrant, "FAQ_Multilingue", mock_model, "")
        assert results == []
        mock_qdrant.query_points.assert_not_called()
