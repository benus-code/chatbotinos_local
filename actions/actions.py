"""Rasa custom actions.

FR: Actions personnalisées pour router intelligemment les requêtes utilisateur.
RU: Пользовательские действия для интеллектуальной маршрутизации запросов.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

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
            dispatcher.utter_message(response="utter_faq_deterministic")
            return []

        if confidence >= INTENT_LOW_CONF:
            route = "B_SIMULATED_QDRANT"
            LOGGER.info(
                "[Intention détectée: %s] | [Score de confiance: %.4f] | [Route choisie: %s]",
                intent_name,
                confidence,
                route,
            )
            LOGGER.info("Interrogation Qdrant simulée | seuil RAG_MIN_SCORE=%.2f", RAG_MIN_SCORE)
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
