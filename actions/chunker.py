"""Intelligent chunker for Russian legislative documents.

Supports structured parsing of chapters (Глава) and articles (Статья),
with a sliding-window fallback for non-legislative text.

Only standard-library dependencies: re, typing.
"""

from __future__ import annotations

import re
from typing import Dict, List, Tuple


# ---------------------------------------------------------------------------
# Type alias
# ---------------------------------------------------------------------------

Chunk = Dict[str, object]  # {"text": str, "metadata": {"article": str, "chapter": str, "type": str}}


# ---------------------------------------------------------------------------
# Compiled patterns
# ---------------------------------------------------------------------------

# Matches chapter headers at the start of a line, e.g.:
#   "Глава 1. Общие положения"  /  "ГЛАВА III"
_CHAPTER_RE = re.compile(
    r"^((?:Глава|ГЛАВА)\s+[^\n]+)",
    re.MULTILINE,
)

# Matches article headers at the start of a line, e.g.:
#   "Статья 5. Права граждан"  /  "СТАТЬЯ 12."
_ARTICLE_RE = re.compile(
    r"^((?:Статья|СТАТЬЯ)\s+[^\n]+)",
    re.MULTILINE,
)

# Used to detect whether a document is legislative at all.
_LEGISLATIVE_SIGNAL_RE = re.compile(
    r"(?:Статья|СТАТЬЯ)\s+\d",
    re.MULTILINE,
)


class LegislativeChunker:
    """Chunk Russian legislative documents into structured segments.

    Legislative documents are split hierarchically:
        Глава  →  Статья  →  chunk

    Non-legislative documents fall back to a sliding-window over words:
        window = 512 words, overlap = 50 words.
    """

    FALLBACK_WINDOW: int = 512
    FALLBACK_OVERLAP: int = 50

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk(self, text: str) -> List[Chunk]:
        """Return a list of chunks for *text*.

        Each chunk is::

            {
                "text": str,
                "metadata": {
                    "article": str,   # e.g. "Статья 5. Права граждан"
                    "chapter": str,   # e.g. "Глава 1. Общие положения"
                    "type": "legislative" | "paragraph",
                },
            }
        """
        text = text.strip()
        if not text:
            return []

        if self._is_legislative(text):
            return self._chunk_legislative(text)
        return self._chunk_fallback(text)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def _is_legislative(self, text: str) -> bool:
        """Return True when the text contains at least one Статья header."""
        return bool(_LEGISLATIVE_SIGNAL_RE.search(text))

    # ------------------------------------------------------------------
    # Legislative path
    # ------------------------------------------------------------------

    def _chunk_legislative(self, text: str) -> List[Chunk]:
        chunks: List[Chunk] = []

        for chapter_title, chapter_body in self._split_by_chapters(text):
            for article_title, article_body in self._split_by_articles(chapter_body):
                body = article_body.strip()
                if not body and not article_title:
                    continue
                # Include the article header in the chunk text so it is
                # self-contained when retrieved from Qdrant.
                chunk_text = f"{article_title}\n{body}".strip() if article_title else body
                chunks.append(self._make_chunk(chunk_text, article_title, chapter_title))

        return chunks

    # ------------------------------------------------------------------
    # Splitting helpers
    # ------------------------------------------------------------------

    def _split_by_chapters(self, text: str) -> List[Tuple[str, str]]:
        """Return [(chapter_title, chapter_body), ...].

        Text before the first chapter is returned with an empty title.
        If no chapters are found, the whole text is returned with an empty title.
        """
        return self._split_with_pattern(_CHAPTER_RE, text)

    def _split_by_articles(self, text: str) -> List[Tuple[str, str]]:
        """Return [(article_title, article_body), ...].

        Text before the first article is returned with an empty title.
        If no articles are found, the whole text is returned with an empty title.
        """
        return self._split_with_pattern(_ARTICLE_RE, text)

    @staticmethod
    def _split_with_pattern(pattern: re.Pattern, text: str) -> List[Tuple[str, str]]:
        """Generic split: divide *text* on *pattern* capturing group.

        ``re.split`` with a single capturing group produces:
            [before, header1, body1, header2, body2, ...]
        """
        parts = pattern.split(text)
        # parts[0]  — text before the first match (preamble / intro)
        # parts[1]  — first header (captured group)
        # parts[2]  — body after first header
        # ...

        sections: List[Tuple[str, str]] = []

        # Preamble (may be empty)
        preamble = parts[0].strip()
        if preamble:
            sections.append(("", preamble))

        # Paired (header, body) groups
        i = 1
        while i + 1 < len(parts):
            header = parts[i].strip()
            body = parts[i + 1].strip()
            sections.append((header, body))
            i += 2

        # Edge case: pattern never matched → treat full text as one section
        if not sections:
            sections.append(("", text.strip()))

        return sections

    # ------------------------------------------------------------------
    # Fallback path (non-legislative text)
    # ------------------------------------------------------------------

    def _chunk_fallback(self, text: str) -> List[Chunk]:
        """Sliding-window chunking over words (no external tokenizer)."""
        words = text.split()
        if not words:
            return []

        chunks: List[Chunk] = []
        window = self.FALLBACK_WINDOW
        overlap = self.FALLBACK_OVERLAP
        start = 0

        while start < len(words):
            end = min(start + window, len(words))
            chunk_text = " ".join(words[start:end])
            chunks.append(self._make_chunk(chunk_text, "", "", kind="paragraph"))
            if end == len(words):
                break
            start += window - overlap

        return chunks

    # ------------------------------------------------------------------
    # Chunk factory
    # ------------------------------------------------------------------

    @staticmethod
    def _make_chunk(
        text: str,
        article: str,
        chapter: str,
        kind: str = "legislative",
    ) -> Chunk:
        return {
            "text": text,
            "metadata": {
                "article": article,
                "chapter": chapter,
                "type": kind,
            },
        }
