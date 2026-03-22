"""Tests for FAQ indexation utilities.

FR: Tests unitaires pour les fonctions d'indexation FAQ.
RU: Юнит-тесты для функций индексации FAQ.
"""

import pytest
from unittest.mock import patch, MagicMock

SAMPLE_FAQ = """\
?? VISA
Q-"Quel est le délai pour obtenir le visa ?"
R- Il faut compter entre 30 et 40 jours.
Q-"Quels documents sont requis ?"
R- Un passeport valide et le baccalauréat.
Q-"Comment prolonger le visa ?"
R-"""


class TestLoadAndSplitFaq:
    @pytest.fixture
    def sample_faq_file(self, tmp_path):
        faq_file = tmp_path / "FAQ.txt"
        faq_file.write_text(SAMPLE_FAQ, encoding="utf-8")
        return str(faq_file)

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

    def test_raises_for_missing_file(self):
        with patch("indexation.model", MagicMock()), patch("indexation.client", MagicMock()):
            from indexation import load_and_split_faq
            with pytest.raises(FileNotFoundError):
                load_and_split_faq("/nonexistent/path/FAQ.txt")


class TestParseFaqBlock:
    @pytest.fixture(autouse=True)
    def patch_module(self):
        with patch("indexation.model", MagicMock()), patch("indexation.client", MagicMock()):
            yield

    def test_extracts_question_and_answer(self):
        from indexation import parse_faq_block
        block = 'Q-"What is the delay?"\nR- About 30 days.'
        result = parse_faq_block(block)
        assert result["question"] == "What is the delay?"
        assert "30 days" in result["answer"]

    def test_empty_answer(self):
        from indexation import parse_faq_block
        block = 'Q-"Question?"\nR-'
        result = parse_faq_block(block)
        assert result["answer"] == ""

    def test_missing_question_marker(self):
        from indexation import parse_faq_block
        block = "No question marker here\nR- Some answer."
        result = parse_faq_block(block)
        assert result["question"] == ""


class TestBuildQdrantPoints:
    @pytest.fixture
    def mock_model(self):
        m = MagicMock()
        m.encode.return_value = [0.1] * 1024
        return m

    def build(self, items, model, source):
        with patch("indexation.model", MagicMock()), patch("indexation.client", MagicMock()):
            from indexation import build_qdrant_points
            return build_qdrant_points(items, model, source)

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


class TestSearchFaq:
    @pytest.fixture
    def mock_qdrant(self):
        m = MagicMock()
        mock_point = MagicMock()
        m.query_points.return_value.points = [mock_point]
        return m

    @pytest.fixture
    def mock_model(self):
        m = MagicMock()
        m.encode.return_value = [0.1] * 1024
        return m

    def search(self, qdrant, collection, model, question, limit=3):
        with patch("indexation.model", MagicMock()), patch("indexation.client", MagicMock()):
            from indexation import search_faq
            return search_faq(qdrant, collection, model, question, limit)

    def test_returns_points_for_valid_query(self, mock_qdrant, mock_model):
        results = self.search(mock_qdrant, "FAQ_Multilingue", mock_model, "visa étudiant")
        assert len(results) > 0

    def test_empty_query_returns_empty_list(self, mock_qdrant, mock_model):
        results = self.search(mock_qdrant, "FAQ_Multilingue", mock_model, "")
        assert results == []
