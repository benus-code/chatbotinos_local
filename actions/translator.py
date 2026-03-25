"""Translation helpers for the RU↔FR RAG pipeline.

Two public classes:
    RuFrTranslator  — Russian → French  (Helsinki-NLP/opus-mt-ru-fr)
    FrToRuTranslator — French → Russian  (Helsinki-NLP/opus-mt-fr-ru)

Design principles:
- Lazy model loading: the MarianMT model is loaded on first use, not at import.
- In-memory cache: hash(text) → translated string; avoids re-translating the
  same chunk on every request.
- Long-text chunking: texts longer than ~400 words are split on sentence
  boundaries (\\n, .) before translation, then reassembled; this prevents
  MarianMT's 512-token hard limit from silently truncating content.
- Fallback: any exception during translation returns the original text so the
  RAG pipeline never crashes due to a translation error.

No external dependencies beyond `transformers`, `sentencepiece`, `sacremoses`,
and `torch` (all listed in requirements-actions.txt).
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

LOGGER = logging.getLogger(__name__)

# Maximum number of *words* per translation segment.
# MarianMT max is 512 sub-word tokens; 400 words gives a comfortable margin.
_MAX_SEGMENT_WORDS = 400


class _BaseTranslator:
    """Shared logic for MarianMT-based translators."""

    MODEL_NAME: str = ""  # Must be overridden in subclasses

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._load_failed: bool = False
        # Cache: hash(original_text) → translated_text
        self._cache: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Lazy loading
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Load tokenizer and model on first call. Sets _load_failed on error."""
        if self._tokenizer is not None:
            return  # Already loaded
        if self._load_failed:
            raise RuntimeError(f"Model {self.MODEL_NAME} previously failed to load.")

        try:
            from transformers import MarianMTModel, MarianTokenizer  # type: ignore

            LOGGER.info("Loading translation model %s …", self.MODEL_NAME)
            self._tokenizer = MarianTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = MarianMTModel.from_pretrained(self.MODEL_NAME)
            LOGGER.info("Translation model %s ready.", self.MODEL_NAME)
        except Exception as exc:
            self._load_failed = True
            raise RuntimeError(
                f"Cannot load {self.MODEL_NAME}: {exc}"
            ) from exc

    # ------------------------------------------------------------------
    # Sentence splitting
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_segments(text: str, max_words: int = _MAX_SEGMENT_WORDS) -> List[str]:
        """Split *text* into segments of at most *max_words* words.

        Splits on sentence boundaries (newlines and periods) to keep segments
        semantically coherent, then groups them greedily until the word budget
        is exhausted.
        """
        # Split on newlines and sentence-ending periods while keeping the
        # delimiter attached to the preceding fragment.
        raw_fragments = re.split(r"(\n+|(?<=\.)\s+)", text)

        # Merge the split-off delimiters back into the preceding fragment.
        fragments: List[str] = []
        for frag in raw_fragments:
            if not frag:
                continue
            if fragments and re.fullmatch(r"[\n\s]+", frag):
                fragments[-1] += frag
            else:
                fragments.append(frag)

        segments: List[str] = []
        current_parts: List[str] = []
        current_words = 0

        for frag in fragments:
            frag_words_list = frag.split()
            frag_word_count = len(frag_words_list)

            if current_words + frag_word_count > max_words and current_parts:
                segments.append("".join(current_parts))
                current_parts = []
                current_words = 0

            if frag_word_count > max_words:
                # Fragment itself exceeds the limit: slice into word-based chunks.
                for i in range(0, frag_word_count, max_words):
                    segments.append(" ".join(frag_words_list[i : i + max_words]))
            else:
                current_parts.append(frag)
                current_words += frag_word_count

        if current_parts:
            segments.append("".join(current_parts))

        return segments if segments else [text]

    # ------------------------------------------------------------------
    # Low-level translation
    # ------------------------------------------------------------------

    def _translate_segment(self, text: str) -> str:
        """Translate a single segment using the loaded MarianMT model."""
        import html as _html  # built-in

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        generated = self._model.generate(**inputs)
        return _html.unescape(
            self._tokenizer.decode(generated[0], skip_special_tokens=True)
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def translate(self, text: str) -> str:
        """Translate *text*, returning the original on any error.

        Results are cached in memory by ``hash(text)``.
        """
        if not text or not text.strip():
            return text

        key = hash(text)
        if key in self._cache:
            return self._cache[key]

        try:
            self._load()
            segments = self._split_into_segments(text)
            translated_parts = [self._translate_segment(seg) for seg in segments]
            result = " ".join(part.strip() for part in translated_parts if part.strip())
            self._cache[key] = result
            return result
        except Exception:
            LOGGER.warning(
                "Translation failed for model %s — returning original text.",
                self.MODEL_NAME,
                exc_info=True,
            )
            return text

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Translate each string in *texts* independently.

        Returns a list of the same length; failed translations are replaced by
        the original string (no exception raised).
        """
        return [self.translate(t) for t in texts]


# ---------------------------------------------------------------------------
# Concrete translators
# ---------------------------------------------------------------------------

class RuFrTranslator(_BaseTranslator):
    """Translate Russian text to French using Helsinki-NLP/opus-mt-ru-fr."""

    MODEL_NAME = "Helsinki-NLP/opus-mt-ru-fr"


class FrToRuTranslator(_BaseTranslator):
    """Translate French text to Russian using Helsinki-NLP/opus-mt-fr-ru."""

    MODEL_NAME = "Helsinki-NLP/opus-mt-fr-ru"
