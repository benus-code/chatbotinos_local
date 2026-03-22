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
)

LOGGER = logging.getLogger(__name__)

RAG_MIN_SCORE = float(os.getenv("RAG_MIN_SCORE", "0.35"))


class ActionHybridRouter(Action):
    """Search Qdrant and return the best matching answer."""

    def name(self) -> Text:
        return "action_hybrid_router"

    @staticmethod
    def _format_result(r: Any) -> str:
        doc_type = r.payload.get("type", "")
        content = r.payload.get("content", "")

        if doc_type == "faq":
            source = r.payload.get("source_file", "FAQ")
            return f"{content}\n\n_Source : {source}_"

        source = r.payload.get("source", "corpus")
        page = r.payload.get("page")
        source_label = f"{source} — p. {page}" if page else source
        return f"{content}\n\n_Source : {source_label}_"

    def _rag_lookup(self, user_text: str) -> Optional[str]:
        try:
            results = search_faq(
                qdrant_client, collection_name, embedding_model, user_text, limit=3
            )
            if not results:
                return None

            def _is_useful(r: Any) -> bool:
                content = r.payload.get("content", "").strip()
                if len(content) < 30:
                    return False
                if r.payload.get("type") == "faq":
                    answer = content.split("Réponse:", 1)[-1].strip() if "Réponse:" in content else ""
                    return len(answer) > 5 and not answer.startswith("??")
                return True

            usable = [r for r in results if r.score >= RAG_MIN_SCORE and _is_useful(r)]
            if not usable:
                LOGGER.info("No usable result above score %.2f", RAG_MIN_SCORE)
                return None

            top = usable[0]
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
