"""Shared pytest fixtures for actions tests.

All heavy dependencies (Qdrant, SentenceTransformer) are mocked so tests
run without a GPU or a running Qdrant instance.

FR: Fixtures pytest partagées — mocks Qdrant et modèle d'embedding.
RU: Общие фикстуры pytest — моки Qdrant и модели эмбеддингов.
"""

from __future__ import annotations

from typing import List
from unittest.mock import MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Embedding model mock
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_model() -> MagicMock:
    """Return a MagicMock that mimics SentenceTransformer.encode()."""
    m = MagicMock()
    m.encode.return_value = np.array([0.1] * 1024)
    return m


# ---------------------------------------------------------------------------
# Qdrant client mock
# ---------------------------------------------------------------------------

def _make_fake_point(score: float, content: str, doc_type: str = "faq") -> MagicMock:
    point = MagicMock()
    point.score = score
    point.payload = {
        "content": content,
        "type": doc_type,
        "source_file": "FAQ.txt",
    }
    return point


@pytest.fixture
def mock_qdrant() -> MagicMock:
    """Return a MagicMock that mimics QdrantClient.query_points()."""
    client = MagicMock()
    result = MagicMock()
    result.points = [
        _make_fake_point(
            0.72,
            'Question: Comment obtenir le visa ?\nRéponse: Contacter le département international.',
        ),
        _make_fake_point(
            0.65,
            'Question: Quels documents pour le visa ?\nRéponse: Passeport valide 18 mois.',
        ),
    ]
    client.query_points.return_value = result
    return client


# ---------------------------------------------------------------------------
# Sample FAQ file fixture
# ---------------------------------------------------------------------------

_SAMPLE_FAQ = """\
Q-"Quel est le délai pour obtenir le visa ?"
R- Il faut compter entre 30 et 40 jours.
Q-"Quels documents sont requis ?"
R- Un passeport valide et le baccalauréat.
Q-"Comment prolonger le visa ?"
R-
"""


@pytest.fixture
def sample_faq_file(tmp_path):
    """Write a small FAQ.txt into a temporary directory and return its path."""
    faq = tmp_path / "FAQ.txt"
    faq.write_text(_SAMPLE_FAQ, encoding="utf-8")
    return str(faq)


# ---------------------------------------------------------------------------
# Sample PDF chunks fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_pdf_chunks() -> List[dict]:
    """Return a list of pre-built PDF chunks (no actual PDF needed)."""
    return [
        {
            "content": "Les étudiants ont le droit de résider à la cité universitaire.",
            "source": "reglement.pdf",
            "page": None,
            "type": "pdf",
            "article_num": "3.1",
            "section_num": "3",
            "section_titre": "Droits des étudiants",
        },
        {
            "content": "Il est interdit de fumer dans les espaces communs.",
            "source": "reglement.pdf",
            "page": None,
            "type": "pdf",
            "article_num": "4.2",
            "section_num": "4",
            "section_titre": "Obligations",
        },
    ]
