"""Rasa custom actions.

FR: Actions personnalisées pour router intelligemment les requêtes utilisateur.
RU: Пользовательские действия для интеллектуальной маршрутизации запросов.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from indexation import (
    collection_name,
    client as qdrant_client,
    model as embedding_model,
    search_faq,
)

LOGGER = logging.getLogger(__name__)


# FR: Seuils de décision pilotés via variables d'environnement.
# RU: Пороги принятия решения, управляемые через переменные окружения.
INTENT_HIGH_CONF = float(os.getenv("INTENT_HIGH_CONF", "0.75"))
INTENT_LOW_CONF = float(os.getenv("INTENT_LOW_CONF", "0.45"))
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.55"))


class ActionHybridRouter(Action):
    """Route user requests based on intent confidence.

    FR:
    - Cas A: confiance > HIGH => réponse déterministe (Rasa standard).
    - Cas B: LOW <= confiance <= HIGH => simulation d'interrogation Qdrant.
    - Cas C: confiance < LOW => fallback vers `utter_default`.

    RU:
    - Случай A: уверенность > HIGH => детерминированный ответ (стандарт Rasa).
    - Случай B: LOW <= уверенность <= HIGH => симуляция запроса к Qdrant.
    - Случай C: уверенность < LOW => fallback в `utter_default`.
    """

    def name(self) -> Text:
        return "action_hybrid_router"

    def _rag_lookup(self, user_text: str) -> Optional[str]:
        """Query Qdrant and return formatted content, or None if below threshold.

        FR: Interroge Qdrant et retourne le contenu formaté, ou None si score insuffisant.
        RU: Запрашивает Qdrant и возвращает отформатированный контент, или None если балл ниже порога.
        """
        try:
            results = search_faq(
                qdrant_client, collection_name, embedding_model, user_text, limit=1
            )
            if results and results[0].score >= RAG_MIN_SCORE:
                r = results[0]
                source = r.payload.get("source_file") or r.payload.get("source", "corpus")
                LOGGER.info("Qdrant result: score=%.4f source=%s", r.score, source)
                return f"{r.payload['content']}\n\n_Source : {source}_"
        except Exception:
            LOGGER.exception("Qdrant lookup failed — graceful degradation")
        return None

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        del domain

        latest_intent = tracker.latest_message.get("intent") or {}
        intent_name = latest_intent.get("name", "unknown")
        confidence = float(latest_intent.get("confidence", 0.0))
        user_text = tracker.latest_message.get("text", "")

        # FR/RU: Journalisation standardisée pour audit rapide en console.
        route = "UNSET"

        if confidence > INTENT_HIGH_CONF:
            route = "A_DETERMINISTIC_RASA"
            LOGGER.info(
                "[Intention détectée: %s] | [Score de confiance: %.4f] | [Route choisie: %s]",
                intent_name,
                confidence,
                route,
            )
            rag_answer = self._rag_lookup(user_text)
            if rag_answer:
                dispatcher.utter_message(text=rag_answer)
            else:
                dispatcher.utter_message(response="utter_faq_deterministic")
            return []

        if confidence >= INTENT_LOW_CONF:
            route = "B_QDRANT_RAG"
            LOGGER.info(
                "[Intention détectée: %s] | [Score de confiance: %.4f] | [Route choisie: %s]",
                intent_name,
                confidence,
                route,
            )
            LOGGER.info("Interrogation Qdrant | seuil RAG_MIN_SCORE=%.2f", RAG_MIN_SCORE)
            rag_answer = self._rag_lookup(user_text)
            if rag_answer:
                dispatcher.utter_message(text=rag_answer)
            else:
                dispatcher.utter_message(response="utter_qdrant_simulated")
            return []

        route = "C_FALLBACK"
        LOGGER.info(
            "[Intention détectée: %s] | [Score de confiance: %.4f] | [Route choisie: %s]",
            intent_name,
            confidence,
            route,
        )
        dispatcher.utter_message(response="utter_default")
        return []
