"""Rasa custom actions.

FR: Actions personnalisées pour orchestrer la recherche FAQ dans Qdrant.
RU: Пользовательские действия для поиска FAQ в Qdrant.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from .indexation import client, collection_name, model, search_faq

LOGGER = logging.getLogger(__name__)

INTENT_HIGH_CONF = float(os.getenv("INTENT_HIGH_CONF", "0.80"))
INTENT_LOW_CONF = float(os.getenv("INTENT_LOW_CONF", "0.45"))
MIN_RELEVANCE_SCORE = float(os.getenv("MIN_RELEVANCE_SCORE", "0.50"))

FAQ_DIRECT_RESPONSES = {
    "visa": "utter_visa_documents",
    "invitation": "utter_invitation_procedure",
    "validité": "utter_visa_validity",
    "validite": "utter_visa_validity",
}


class ActionSearchFAQ(Action):
    """Handle FAQ lookup and return a bilingual answer.

    FR: Effectue une recherche vectorielle et répond en format FR/RU.
    RU: Выполняет векторный поиск и отвечает в формате FR/RU.
    """

    def name(self) -> Text:
        return "action_search_faq"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        """Process user message, query knowledge base, and respond safely.

        FR: Récupère la question utilisateur, cherche la meilleure correspondance,
        puis envoie une réponse formatée avec gestion explicite des erreurs.

        RU: Получает вопрос пользователя, ищет лучшее совпадение,
        затем отправляет форматированный ответ с явной обработкой ошибок.
        """
        user_message = tracker.latest_message.get("text")
        if not user_message:
            LOGGER.warning("Empty user message received in action_search_faq.")
            dispatcher.utter_message(text="Désolé, je n'ai pas reçu votre question. Pouvez-vous reformuler ?")
            return []

        latest_intent = tracker.latest_message.get("intent", {})
        intent_name = latest_intent.get("name")
        intent_confidence = float(latest_intent.get("confidence") or 0.0)

        LOGGER.info(
            "Intent detected: %s (confidence=%.3f, high=%.2f, low=%.2f)",
            intent_name,
            intent_confidence,
            INTENT_HIGH_CONF,
            INTENT_LOW_CONF,
        )

        if intent_name == "demander_faq" and intent_confidence >= INTENT_HIGH_CONF:
            lower_message = user_message.lower()
            for keyword, response_name in FAQ_DIRECT_RESPONSES.items():
                if keyword in lower_message and response_name in domain.get("responses", {}):
                    dispatcher.utter_message(response=response_name)
                    LOGGER.info("Direct domain response sent: %s", response_name)
                    return []

        try:
            responses = search_faq(
                qdrant_client=client,
                collection=collection_name,
                embedding_model=model,
                user_question=user_message,
                limit=1,
            )
        except Exception:
            LOGGER.exception("FAQ search failed for message: %s", user_message)
            dispatcher.utter_message(
                text="Désolé, une erreur technique est survenue pendant la recherche. Merci de réessayer."
            )
            return []

        if responses and responses[0].score > MIN_RELEVANCE_SCORE:
            payload = responses[0].payload or {}

            # FR: On priorise 'content' puis fallback legacy 'texte_fr'.
            # RU: Приоритет у 'content', затем legacy-ключ 'texte_fr'.
            french_text = payload.get("content") or payload.get("texte_fr") or "Texte non trouvé"
            russian_text = payload.get("texte_ru", "Version russe non disponible")
            source = payload.get("source", "#")
            page = payload.get("page", "?")

            final_message = (
                f"🇫🇷 **Français :**\n{french_text}\n\n"
                f"🇷🇺 **Русский :**\n{russian_text}\n\n"
                f"📂 **Source :** [Consulter le document (Page {page})]({source})"
            )

            dispatcher.utter_message(text=final_message)
            LOGGER.info("FAQ response sent with score %.4f", responses[0].score)
            return []

        if intent_confidence <= INTENT_LOW_CONF:
            LOGGER.info(
                "Fallback because intent confidence is too low (%.3f <= %.2f)",
                intent_confidence,
                INTENT_LOW_CONF,
            )
        else:
            LOGGER.info(
                "Fallback because retrieval score is below threshold %.2f", MIN_RELEVANCE_SCORE
            )

        dispatcher.utter_message(response="utter_default")
        return []
