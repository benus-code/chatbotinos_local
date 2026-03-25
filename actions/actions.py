"""Rasa custom actions — RAG lookup with RU↔FR translation.

Pipeline:
    1. User question arrives in French.
    2. FrToRuTranslator converts it to Russian.
    3. The Russian question is embedded and searched in Qdrant.
    4. Each retrieved Russian chunk is translated to French by RuFrTranslator.
    5. The response is formatted with all unique source file names.
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
from translator import FrToRuTranslator, RuFrTranslator

LOGGER = logging.getLogger(__name__)

# Score threshold: results below this are discarded (env-overridable).
RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.55"))

# ---------------------------------------------------------------------------
# Module-level translator singletons (lazy-initialised).
# Rasa instantiates a new Action object per request; keeping the translators
# at module level means the heavy MarianMT models are loaded only once.
# ---------------------------------------------------------------------------

_ru_fr_translator: Optional[RuFrTranslator] = None
_fr_ru_translator: Optional[FrToRuTranslator] = None


def _get_ru_fr() -> RuFrTranslator:
    global _ru_fr_translator
    if _ru_fr_translator is None:
        _ru_fr_translator = RuFrTranslator()
    return _ru_fr_translator


def _get_fr_ru() -> FrToRuTranslator:
    global _fr_ru_translator
    if _fr_ru_translator is None:
        _fr_ru_translator = FrToRuTranslator()
    return _fr_ru_translator


# ---------------------------------------------------------------------------
# Action
# ---------------------------------------------------------------------------

class ActionHybridRouter(Action):
    """Translate the user's French question to Russian, search Qdrant, and
    return the results translated back to French with source attribution."""

    def name(self) -> Text:
        return "action_hybrid_router"

    # ------------------------------------------------------------------
    # Response formatting
    # ------------------------------------------------------------------

    @staticmethod
    def _format_translated_results(
        translated_chunks: List[str],
        sources: List[str],
    ) -> str:
        """Build the final French response with source attribution."""
        body = "\n\n".join(chunk for chunk in translated_chunks if chunk.strip())

        if not sources:
            return f"D'après les documents consultés :\n{body}"

        if len(sources) == 1:
            source_block = f"Source : {sources[0]}"
        else:
            source_lines = "\n".join(f"- {s}" for s in sources)
            source_block = f"Sources :\n{source_lines}"

        return f"D'après les documents consultés :\n{body}\n\n{source_block}"

    # ------------------------------------------------------------------
    # RAG lookup
    # ------------------------------------------------------------------

    def _rag_lookup(self, ru_question: str) -> Optional[str]:
        """Search Qdrant with a Russian question and return a French response.

        Returns None when no result meets the score threshold so the caller
        can fall back to ``utter_default``.
        """
        try:
            results = search_faq(
                qdrant_client,
                collection_name,
                embedding_model,
                ru_question,
                limit=3,
            )
            if not results:
                return None

            usable = [r for r in results if r.score >= RAG_MIN_SCORE]
            if not usable:
                LOGGER.info(
                    "No result above score threshold %.2f (best was %.4f).",
                    RAG_MIN_SCORE,
                    results[0].score,
                )
                return None

            translated_chunks: List[str] = []
            sources: List[str] = []

            for r in usable:
                LOGGER.info(
                    "RAG hit: score=%.4f type=%s",
                    r.score,
                    r.payload.get("type", "?"),
                )
                content_ru = r.payload.get("content", "").strip()
                if not content_ru:
                    continue

                content_fr = _get_ru_fr().translate(content_ru)
                translated_chunks.append(content_fr)

                # Support both payload key conventions used in the project.
                source = (
                    r.payload.get("source")
                    or r.payload.get("source_file")
                    or ""
                )
                if source and source not in sources:
                    sources.append(source)

            if not translated_chunks:
                return None

            return self._format_translated_results(translated_chunks, sources)

        except Exception:
            LOGGER.exception("Qdrant lookup failed")
            return None

    # ------------------------------------------------------------------
    # Rasa entry point
    # ------------------------------------------------------------------

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:
        del domain

        user_text = tracker.latest_message.get("text", "")
        LOGGER.info("ActionHybridRouter triggered: %r", user_text)

        # Translate the French question to Russian before embedding.
        ru_question = _get_fr_ru().translate(user_text)
        LOGGER.info("Question translated to RU: %r", ru_question)

        answer = self._rag_lookup(ru_question)
        if answer:
            dispatcher.utter_message(text=answer)
        else:
            dispatcher.utter_message(response="utter_default")

        return []
