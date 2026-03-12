"""Utilities for FAQ indexing and semantic retrieval with Qdrant.

Ce module centralise l'indexation FAQ et la recherche sémantique.
Этот модуль централизует индексацию FAQ и семантический поиск.
"""

from __future__ import annotations

import logging
import re
import uuid
from pathlib import Path
from typing import Any, Dict, List

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import PointStruct
from sentence_transformers import SentenceTransformer

LOGGER = logging.getLogger(__name__)

# Configuration Qdrant / Embedding
# Configuration de la connexion Qdrant et du modèle d'embedding.
# Конфигурация подключения к Qdrant и модели эмбеддингов.
client = QdrantClient(host="localhost", port=6333, timeout=60)
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
collection_name = "FAQ_Multilingue"


def load_and_split_faq(file_path: str) -> List[str]:
    """Load FAQ file and split it into Q/R blocks.

    FR: Charge le fichier FAQ et découpe les blocs basés sur le préfixe `Q-`.
    RU: Загружает FAQ-файл и разбивает его на блоки по префиксу `Q-`.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"FAQ file not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        text_to_vectorize = file.read()

    blocks = re.split(r"\n(?=Q-)", text_to_vectorize)
    return [block.strip() for block in blocks if block.strip()]


def parse_faq_block(block: str) -> Dict[str, str]:
    """Extract structured fields from a single FAQ block.

    FR: Extrait question/réponse d'un bloc texte FAQ brut.
    RU: Извлекает вопрос/ответ из сырого текстового FAQ-блока.
    """
    question_match = re.search(r'Q-"(.*?)"', block)
    answer_match = re.search(r"R-(.*)", block, re.DOTALL)

    return {
        "text_to_vectorize": block,
        "question": question_match.group(1) if question_match else "",
        "answer": answer_match.group(1).strip() if answer_match else "",
    }


def build_qdrant_points(
    faq_items: List[Dict[str, str]],
    embedding_model: SentenceTransformer,
    source_file_name: str,
) -> List[PointStruct]:
    """Generate Qdrant points from parsed FAQ entries.

    FR: Construit les points Qdrant (vecteurs + métadonnées) pour chaque entrée FAQ.
    RU: Формирует точки Qdrant (вектора + метаданные) для каждой записи FAQ.
    """
    points: List[PointStruct] = []

    for item in faq_items:
        unique_id = str(uuid.uuid4())
        text_to_vectorize = f"Question: {item['question']} Réponse: {item['answer']}"
        vector = embedding_model.encode(text_to_vectorize).tolist()

        points.append(
            PointStruct(
                id=unique_id,
                vector=vector,
                payload={
                    "content": f"Question: {item['question']}\nRéponse: {item['answer']}",
                    "source_file": source_file_name,
                    "type": "faq",
                },
            )
        )

    return points


def replace_source_points(
    qdrant_client: QdrantClient,
    collection: str,
    source_file_name: str,
    new_points: List[PointStruct],
) -> None:
    """Replace all points from a source file, then insert fresh points.

    FR: Supprime les points existants d'une source puis insère les nouveaux.
    RU: Удаляет существующие точки источника и вставляет новые.
    """
    qdrant_client.delete(
        collection_name=collection,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="source_file",
                    match=models.MatchValue(value=source_file_name),
                )
            ]
        ),
    )
    qdrant_client.upsert(collection_name=collection, points=new_points)


def search_faq(
    qdrant_client: QdrantClient,
    collection: str,
    embedding_model: SentenceTransformer,
    user_question: str,
    limit: int = 3,
) -> List[Any]:
    """Search semantic FAQ matches in Qdrant.

    FR: Recherche les meilleures réponses FAQ selon la similarité vectorielle.
    RU: Ищет лучшие ответы FAQ по векторному сходству.
    """
    if not user_question:
        return []

    question_vector = embedding_model.encode(user_question).tolist()
    result = qdrant_client.query_points(
        collection_name=collection,
        query=question_vector,
        limit=limit,
    )
    return result.points


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        # FR: Crée la collection si elle n'existe pas.
        # RU: Создает коллекцию, если она не существует.
        if not client.collection_exists(collection_name=collection_name):
            client.create_collection(
                collection_name=collection_name,
                vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
            )
            LOGGER.info("Collection '%s' created.", collection_name)

        source_file_name = "FAQ.txt"
        content_blocks = load_and_split_faq(source_file_name)
        structured_faq = [parse_faq_block(block) for block in content_blocks]

        points = build_qdrant_points(structured_faq, model, source_file_name)
        replace_source_points(
            qdrant_client=client,
            collection=collection_name,
            source_file_name=source_file_name,
            new_points=points,
        )
        LOGGER.info("FAQ data successfully indexed into '%s'.", collection_name)

        test_question = "Je veux savoir comment avoir l'invitation"
        search_results = search_faq(client, collection_name, model, test_question)

        LOGGER.info("Search test for question: %s", test_question)
        for rank, result_item in enumerate(search_results, start=1):
            LOGGER.info("Rank %d (score %.4f): %s", rank, result_item.score, result_item.payload.get("content"))

    except Exception:
        LOGGER.exception("Indexation script failed.")
        raise
