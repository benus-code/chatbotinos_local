"""Rasa custom actions — RAG lookup with RU↔FR translation.

Pipeline
--------
1. User question arrives in French.
2. FrToRuTranslator converts it to Russian (to match the Russian corpus).
3. The Russian question is embedded with "query: " prefix and searched in Qdrant
   (collection "documents", top_k=3).
4. All returned scores are logged; results below RAG_MIN_SCORE are discarded.
5. Each retained Russian chunk is translated to French by RuFrTranslator.
6. The response is formatted with per-chunk article + source metadata.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.executor import CollectingDispatcher

from indexation import (
    client as qdrant_client,
    model as embedding_model,
    search_faq,
)
from translator import FrToRuTranslator, RuFrTranslator

LOGGER = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration (env-overridable)
# ---------------------------------------------------------------------------

# Qdrant collection that holds the Russian legislative documents.
RAG_COLLECTION = os.getenv("RAG_COLLECTION", "documents")

# Results with a cosine score below this threshold are discarded.
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
    return the results translated back to French with article + source metadata."""

    def name(self) -> Text:
        return "action_hybrid_router"

    # ------------------------------------------------------------------
    # Response formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_chunk_block(rank: int, content_fr: str, article: str, source: str) -> str:
        """Format a single translated chunk with its rank, article, and source.

        Example output:
            [1] Article : Статья 5. Права студентов | Source : loi.pdf
            Le texte traduit du chunk...
        """
        header_parts: List[str] = []
        if article:
            header_parts.append(f"Article : {article}")
        if source:
            header_parts.append(f"Source : {source}")
        meta = " | ".join(header_parts)
        header = f"[{rank}] {meta}".strip() if meta else f"[{rank}]"
        return f"{header}\n{content_fr.strip()}"

    @staticmethod
    def _format_response(blocks: List[str], all_sources: List[str]) -> str:
        """Assemble the full French response from translated blocks and sources.

        Single source:
            D'après les documents consultés :
            [1] ...
            Source : fichier.pdf

        Multiple sources:
            D'après les documents consultés :
            [1] ...
            [2] ...
            Sources :
            - fichier1.pdf
            - fichier2.pdf
        """
        body = "\n\n".join(b for b in blocks if b.strip())
        intro = "D'après les documents consultés :"

        if not all_sources:
            return f"{intro}\n{body}"

        if len(all_sources) == 1:
            source_block = f"Source : {all_sources[0]}"
        else:
            source_lines = "\n".join(f"- {s}" for s in all_sources)
            source_block = f"Sources :\n{source_lines}"

        return f"{intro}\n\n{body}\n\n{source_block}"

    # ------------------------------------------------------------------
    # RAG lookup
    # ------------------------------------------------------------------

    def _rag_lookup(self, ru_question: str) -> Optional[str]:
        """Search Qdrant with a Russian question; return a formatted French response.

        Logs every returned score, then translates the retained chunks.
        Returns None when no result meets the score threshold.
        """
        try:
            results = search_faq(
                qdrant_client,
                RAG_COLLECTION,
                embedding_model,
                ru_question,
                limit=3,
            )

            if not results:
                LOGGER.info("Qdrant returned no results for collection %r.", RAG_COLLECTION)
                return None

            # --- Log ALL scores for observability ---
            for rank, r in enumerate(results, start=1):
                LOGGER.info(
                    "Qdrant result: rank=%d score=%.4f source=%r article=%r",
                    rank,
                    r.score,
                    r.payload.get("source") or r.payload.get("source_file", "?"),
                    r.payload.get("article", ""),
                )

            # --- Filter by score threshold ---
            usable = [r for r in results if r.score >= RAG_MIN_SCORE]
            if not usable:
                LOGGER.info(
                    "No result above threshold %.2f (best score was %.4f).",
                    RAG_MIN_SCORE,
                    results[0].score,
                )
                return None

            LOGGER.info(
                "Selected %d/%d chunks above threshold %.2f.",
                len(usable),
                len(results),
                RAG_MIN_SCORE,
            )

            # --- Translate and format each retained chunk ---
            blocks: List[str] = []
            all_sources: List[str] = []

            for rank, r in enumerate(usable, start=1):
                content_ru = r.payload.get("content", "").strip()
                if not content_ru:
                    LOGGER.warning("Chunk rank=%d has empty content, skipping.", rank)
                    continue

                LOGGER.info(
                    "Translating chunk rank=%d (score=%.4f): %r …",
                    rank,
                    r.score,
                    content_ru[:120],
                )

                content_fr = _get_ru_fr().translate(content_ru)

                article = r.payload.get("article", "")
                source = r.payload.get("source") or r.payload.get("source_file", "")

                blocks.append(
                    self._format_chunk_block(rank, content_fr, article, source)
                )

                if source and source not in all_sources:
                    all_sources.append(source)

            if not blocks:
                return None

            return self._format_response(blocks, all_sources)

        except Exception:
            LOGGER.exception("Qdrant lookup failed (collection=%r).", RAG_COLLECTION)
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
        LOGGER.info("ActionHybridRouter — question reçue : %r", user_text)

        # Translate the French question to Russian before embedding so it
        # matches the Russian documents stored in Qdrant.
        ru_question = _get_fr_ru().translate(user_text)
        LOGGER.info("Question traduite en RU : %r", ru_question)

        answer = self._rag_lookup(ru_question)
        if answer:
            dispatcher.utter_message(text=answer)
        else:
            dispatcher.utter_message(response="utter_default")

        return []
