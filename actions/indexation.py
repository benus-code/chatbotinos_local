"""Utilities for FAQ indexing and semantic retrieval with Qdrant.

Ce module centralise l'indexation FAQ et la recherche sémantique.
Этот модуль централизует индексацию FAQ и семантический поиск.
"""

from __future__ import annotations

import logging
import os
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
client = QdrantClient(host=os.getenv("QDRANT_HOST", "qdrant"), port=6333, timeout=60)
# Modèle multilingue : projette le français et le russe dans le même espace vectoriel.
# Une question en français retrouve directement un document en russe sans traduction.
# Многоязычная модель: проецирует французский и русский в одно векторное пространство.
# Французский запрос находит русские документы напрямую, без перевода.
model = SentenceTransformer("intfloat/multilingual-e5-large")
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

    answer = ""
    if answer_match:
        # Strip section headers (?? ...) that bleed into answers when R- is empty
        lines = [l for l in answer_match.group(1).splitlines() if not l.strip().startswith("??")]
        answer = "\n".join(lines).strip()

    return {
        "text_to_vectorize": block,
        "question": question_match.group(1) if question_match else "",
        "answer": answer,
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
        # N'indexe pas les entrées FAQ sans réponse — elles polluent les résultats PDF.
        # Не индексируем FAQ-записи без ответа — они засоряют результаты PDF.
        if len(item.get("answer", "").strip()) < 10:
            LOGGER.debug("FAQ ignorée (réponse vide) : %s", item.get("question", "")[:60])
            continue

        unique_id = str(uuid.uuid4())
        # Préfixe "passage: " requis par multilingual-e5-large pour l'indexation de documents.
        # Префикс "passage: " обязателен для multilingual-e5-large при индексации документов.
        text_to_vectorize = f"passage: Question: {item['question']} Réponse: {item['answer']}"
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
    """Search semantic FAQ matches in Qdrant (all types).

    FR: Recherche les meilleures réponses selon la similarité vectorielle, tous types confondus.
    RU: Ищет лучшие ответы по векторному сходству, без фильтрации по типу.
    """
    if not user_question:
        return []

    # Préfixe "query: " pour les requêtes utilisateur — asymétrie E5 obligatoire.
    # Префикс "query: " для запросов пользователя — асимметрия E5 обязательна.
    question_vector = embedding_model.encode(f"query: {user_question}").tolist()
    result = qdrant_client.query_points(
        collection_name=collection,
        query=question_vector,
        limit=limit,
    )
    return result.points


def search_by_type(
    qdrant_client: QdrantClient,
    collection: str,
    embedding_model: SentenceTransformer,
    user_question: str,
    doc_type: str,
    limit: int = 3,
) -> List[Any]:
    """Search semantic matches filtered by document type ('faq' or 'pdf').

    FR: Recherche dans Qdrant en filtrant par type de document.
    Permet de chercher FAQ et PDF séparément pour éviter que le français
    de la FAQ n'écrase systématiquement les chunks russes du PDF.
    RU: Поиск в Qdrant с фильтрацией по типу документа.
    Позволяет искать FAQ и PDF раздельно, чтобы французские FAQ-записи
    не вытесняли русские PDF-чанки.
    """
    if not user_question:
        return []

    question_vector = embedding_model.encode(f"query: {user_question}").tolist()
    result = qdrant_client.query_points(
        collection_name=collection,
        query=question_vector,
        query_filter=models.Filter(
            must=[
                models.FieldCondition(
                    key="type",
                    match=models.MatchValue(value=doc_type),
                )
            ]
        ),
        limit=limit,
    )
    return result.points


def build_qdrant_points_pdf(
    chunks: List[Dict[str, str]],
    embedding_model: SentenceTransformer,
) -> List[PointStruct]:
    """Generate Qdrant points from PDF chunks produced by pdf_chunker.py.

    FR: Construit les points Qdrant pour des chunks PDF (clé 'source' au lieu de 'source_file').
    RU: Формирует точки Qdrant для PDF-чанков (ключ 'source' вместо 'source_file').
    """
    points: List[PointStruct] = []

    for chunk in chunks:
        unique_id = str(uuid.uuid4())
        # Préfixe "passage: " requis pour l'indexation des chunks PDF en russe.
        # Префикс "passage: " обязателен при индексации PDF-чанков на русском языке.
        vector = embedding_model.encode(f"passage: {chunk['content']}").tolist()

        payload: Dict[str, Any] = {
            "content": chunk["content"],
            "source": chunk.get("source", ""),
            "page": chunk.get("page"),
            "type": "pdf",
        }
        # Preserve article metadata when present (legal documents)
        for key in ("article_num", "section_num", "section_titre"):
            if chunk.get(key):
                payload[key] = chunk[key]

        points.append(PointStruct(id=unique_id, vector=vector, payload=payload))

    return points


def replace_pdf_source_points(
    qdrant_client: QdrantClient,
    collection: str,
    source_name: str,
    new_points: List[PointStruct],
) -> None:
    """Replace all PDF points from a given source, then insert fresh points.

    FR: Supprime les points PDF existants d'une source puis insère les nouveaux.
    RU: Удаляет существующие PDF-точки источника и вставляет новые.
    """
    qdrant_client.delete(
        collection_name=collection,
        points_selector=models.Filter(
            must=[
                models.FieldCondition(
                    key="source",
                    match=models.MatchValue(value=source_name),
                )
            ]
        ),
    )
    qdrant_client.upsert(collection_name=collection, points=new_points)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    try:
        # Recrée la collection pour garantir la cohérence avec le nouveau modèle d'embedding.
        # Пересоздаём коллекцию, чтобы гарантировать совместимость с новой моделью эмбеддингов.
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)
            LOGGER.info("Collection '%s' supprimée pour recréation.", collection_name)
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=1024, distance=models.Distance.COSINE),
        )
        LOGGER.info("Collection '%s' créée.", collection_name)

        # Chemin absolu basé sur l'emplacement du script — indépendant du répertoire courant.
        # Абсолютный путь на основе расположения скрипта — независимо от рабочей директории.
        _here = Path(__file__).parent
        source_file_name = "FAQ.txt"
        content_blocks = load_and_split_faq(str(_here / source_file_name))
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
