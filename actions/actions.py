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

    @staticmethod
    def _format_result(r: Any) -> str:
        """Format a single Qdrant point into a human-readable response.

        FR: Formate un point Qdrant en réponse lisible selon son type (faq ou pdf).
        RU: Форматирует точку Qdrant в читаемый ответ в зависимости от типа (faq или pdf).
        """
        doc_type = r.payload.get("type", "")
        content = r.payload.get("content", "")

        if doc_type == "faq":
            source = r.payload.get("source_file", "FAQ")
            return f"{content}\n\n_Source : {source}_"

        # PDF chunk
        source = r.payload.get("source", "corpus")
        page = r.payload.get("page")
        source_label = f"{source} — p. {page}" if page else source
        return f"📄 {content}\n\n_Source : {source_label}_"

    def _rag_lookup(self, user_text: str, multi: bool = False) -> Optional[str]:
        """Query Qdrant and return formatted content, or None if below threshold.

        FR: Interroge Qdrant et retourne le contenu formaté (un ou plusieurs résultats).
            Si multi=True, inclut jusqu'à 3 résultats quand les scores sont proches.
        RU: Запрашивает Qdrant и возвращает отформатированный контент (один или несколько).
            При multi=True включает до 3 результатов при близких значениях score.
        """
        try:
            limit = 3 if multi else 1
            results = search_faq(
                qdrant_client, collection_name, embedding_model, user_text, limit=limit
            )
            if not results or results[0].score < RAG_MIN_SCORE:
                return None

            # FR: Filtre qualité — exclut les chunks vides ou trop courts (headers PDF, FAQ sans réponse).
            # RU: Фильтр качества — исключает пустые или слишком короткие чанки.
            def _is_useful(r: Any) -> bool:
                content = r.payload.get("content", "").strip()
                if len(content) < 80:
                    return False
                if r.payload.get("type") == "faq":
                    answer = ""
                    if "Réponse:" in content:
                        answer = content.split("Réponse:", 1)[-1].strip()
                    return len(answer) > 5 and not answer.startswith("??")

                return True

            usable = [r for r in results if r.score >= RAG_MIN_SCORE and _is_useful(r)]
            if not usable:
                return None

            top = usable[0]
            LOGGER.info(
                "Qdrant top result: score=%.4f type=%s",
                top.score,
                top.payload.get("type", "?"),
            )

            if not multi or len(usable) == 1:
                return self._format_result(top)

            # FR: Inclure les résultats supplémentaires dont le score est proche du meilleur.
            # RU: Включать дополнительные результаты с близким значением score.
            close_results = [
                r for r in usable[1:]
                if (top.score - r.score) <= 0.08
            ]
            if not close_results:
                return self._format_result(top)

            parts = [self._format_result(top)]
            for r in close_results:
                parts.append(self._format_result(r))
            return "\n\n---\n\n".join(parts)

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
            rag_answer = self._rag_lookup(user_text, multi=True)
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
            rag_answer = self._rag_lookup(user_text, multi=True)
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
