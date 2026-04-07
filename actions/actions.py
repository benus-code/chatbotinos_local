"""Rasa custom actions — simple RAG lookup."""

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
    search_by_type,
)
from translator import RuFrTranslator

LOGGER = logging.getLogger(__name__)

RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))

# Singleton du traducteur RU→FR — instancié à la première utilisation.
# Синглтон переводчика RU→FR — инициализируется при первом использовании.
_ru_fr_translator: Optional[RuFrTranslator] = None


def _get_ru_fr() -> RuFrTranslator:
    global _ru_fr_translator
    if _ru_fr_translator is None:
        _ru_fr_translator = RuFrTranslator()
    return _ru_fr_translator


class ActionHybridRouter(Action):
    """Search Qdrant and return the best matching answer."""

    _MAX_CONTENT_CHARS = 600

    def name(self) -> Text:
        return "action_hybrid_router"

    @staticmethod
    def _format_result(r: Any) -> str:
        doc_type = r.payload.get("type", "")
        content = r.payload.get("content", "")

        # Truncate long chunks for readability (PDF legal articles can be very long)
        if len(content) > ActionHybridRouter._MAX_CONTENT_CHARS:
            content = content[: ActionHybridRouter._MAX_CONTENT_CHARS].rsplit(" ", 1)[0] + "…"

        if doc_type == "faq":
            # FAQ indexée en français — aucune traduction nécessaire.
            # FAQ индексируется на французском — перевод не нужен.
            source = r.payload.get("source_file", "FAQ")
            return f"{content}\n\n_Source : {source}_"

        # Chunk PDF en russe — traduit en français avant affichage.
        # PDF-чанк на русском — переводим на французский перед отображением.
        content_fr = _get_ru_fr().translate(content)
        source = r.payload.get("source", "corpus")
        page = r.payload.get("page")
        source_label = f"{source} — p. {page}" if page else source
        return f"{content_fr}\n\n_Source : {source_label}_"

    def _rag_lookup(self, user_text: str) -> Optional[str]:
        try:
            # Recherche séparée FAQ et PDF pour éviter que le français de la FAQ
            # n'écrase systématiquement les chunks russes du PDF (biais même-langue).
            # Раздельный поиск по FAQ и PDF, чтобы французские FAQ-записи не вытесняли
            # русские PDF-чанки из-за преимущества одного языка.
            faq_results = search_by_type(
                qdrant_client, collection_name, embedding_model, user_text, "faq", limit=5
            )
            pdf_results = search_by_type(
                qdrant_client, collection_name, embedding_model, user_text, "pdf", limit=5
            )

            def _is_useful(r: Any) -> bool:
                content = r.payload.get("content", "").strip()
                if len(content) < 30:
                    return False
                if r.payload.get("type") == "faq":
                    answer = content.split("Réponse:", 1)[-1].strip() if "Réponse:" in content else ""
                    return len(answer) > 5 and not answer.startswith("??")
                return True

            best_faq = next((r for r in faq_results if r.score >= RAG_MIN_SCORE and _is_useful(r)), None)
            best_pdf = next((r for r in pdf_results if r.score >= RAG_MIN_SCORE and _is_useful(r)), None)

            if not best_faq and not best_pdf:
                LOGGER.info("Aucun résultat utile au-dessus du seuil %.2f", RAG_MIN_SCORE)
                return None

            # Choisit le meilleur résultat entre FAQ et PDF.
            # Si le PDF est dans les 10% du score FAQ, on préfère le PDF (plus spécifique).
            # Выбирает лучший результат между FAQ и PDF.
            # Если PDF в пределах 10% от FAQ, предпочитаем PDF (он более специфичен).
            if best_pdf and best_faq:
                top = best_pdf if best_pdf.score >= best_faq.score - 0.10 else best_faq
            else:
                top = best_pdf or best_faq

            LOGGER.info("RAG result: score=%.4f type=%s", top.score, top.payload.get("type", "?"))
            return self._format_result(top)

        except Exception:
            LOGGER.exception("Qdrant lookup failed")
            return None

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        del domain
        user_text = tracker.latest_message.get("text", "")
        LOGGER.info("ActionRAG triggered for: %s", user_text)

        answer = self._rag_lookup(user_text)
        if answer:
            dispatcher.utter_message(text=answer)
        else:
            dispatcher.utter_message(response="utter_default")
        return []
