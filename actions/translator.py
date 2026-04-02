"""Traducteur MarianMT pour le pipeline RAG — russe vers français.

Ce module fournit RuFrTranslator, utilisé dans actions.py pour traduire
les chunks PDF russes en français avant de les afficher à l'utilisateur.
L'indexation stocke le russe natif ; le modèle multilingual-e5-large assure
la correspondance cross-linguale entre la question française et les documents russes.

Этот модуль предоставляет RuFrTranslator для перевода русских PDF-чанков
на французский язык перед показом пользователю.
При индексации хранится исходный русский текст; модель multilingual-e5-large
обеспечивает кросс-языковое сопоставление французского запроса с русскими документами.
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List

LOGGER = logging.getLogger(__name__)

# Nombre maximum de mots par segment envoyé au tokenizer MarianMT.
# Le tokenizer est limité à 512 sous-tokens ; 400 mots donnent une marge suffisante.
# Максимальное количество слов на сегмент для токенизатора MarianMT.
# Токенизатор ограничен 512 субтокенами; 400 слов — достаточный запас.
_MAX_SEGMENT_WORDS = 400


class _BaseTranslator:
    """Logique partagée entre les traducteurs MarianMT.

    Принципы работы:
    - Отложенная загрузка модели (загружается при первом вызове).
    - Кэш в памяти: hash(текст) → перевод, повторный перевод не выполняется.
    - Разбивка длинных текстов на сегменты по границам предложений.
    - При любой ошибке возвращается исходный текст — pipeline не падает.
    """

    MODEL_NAME: str = ""  # À surcharger dans les sous-classes / Переопределяется в подклассах

    def __init__(self) -> None:
        self._tokenizer = None
        self._model = None
        self._load_failed: bool = False
        # Cache en mémoire : hash(texte_original) → texte_traduit
        # Кэш: hash(оригинал) → перевод
        self._cache: Dict[int, str] = {}

    # ------------------------------------------------------------------
    # Chargement différé / Отложенная загрузка
    # ------------------------------------------------------------------

    def _load(self) -> None:
        """Charge tokenizer et modèle au premier appel.

        Загружает токенизатор и модель при первом вызове.
        """
        if self._tokenizer is not None:
            return
        if self._load_failed:
            raise RuntimeError(f"Le modèle {self.MODEL_NAME} n'a pas pu être chargé.")

        try:
            from transformers import MarianMTModel, MarianTokenizer  # type: ignore

            LOGGER.info("Chargement du modèle de traduction %s …", self.MODEL_NAME)
            self._tokenizer = MarianTokenizer.from_pretrained(self.MODEL_NAME)
            self._model = MarianMTModel.from_pretrained(self.MODEL_NAME)
            LOGGER.info("Modèle %s prêt.", self.MODEL_NAME)
        except Exception as exc:
            self._load_failed = True
            raise RuntimeError(f"Impossible de charger {self.MODEL_NAME}: {exc}") from exc

    # ------------------------------------------------------------------
    # Découpage en segments / Разбивка на сегменты
    # ------------------------------------------------------------------

    @staticmethod
    def _split_into_segments(text: str, max_words: int = _MAX_SEGMENT_WORDS) -> List[str]:
        """Découpe le texte en segments de max_words mots sur les frontières de phrases.

        Разбивает текст на сегменты по max_words слов на границах предложений,
        чтобы не превысить лимит токенизатора.
        """
        raw_fragments = re.split(r"(\n+|(?<=\.)\s+)", text)

        # Réattache les délimiteurs au fragment précédent
        # Возвращаем разделители к предыдущему фрагменту
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
                # Fragment dépasse seul la limite — découpe par mots
                # Фрагмент сам по себе превышает лимит — разбиваем по словам
                for i in range(0, frag_word_count, max_words):
                    segments.append(" ".join(frag_words_list[i : i + max_words]))
            else:
                current_parts.append(frag)
                current_words += frag_word_count

        if current_parts:
            segments.append("".join(current_parts))

        return segments if segments else [text]

    # ------------------------------------------------------------------
    # Traduction d'un segment / Перевод одного сегмента
    # ------------------------------------------------------------------

    def _translate_segment(self, text: str) -> str:
        """Traduit un segment via le modèle MarianMT chargé.

        Переводит один сегмент через загруженную модель MarianMT.
        """
        import html as _html

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
    # API publique / Публичный API
    # ------------------------------------------------------------------

    def translate(self, text: str) -> str:
        """Traduit text ; retourne l'original en cas d'erreur.

        Résultats mis en cache par hash(text) pour éviter les retraductions.

        Переводит текст; при ошибке возвращает оригинал (pipeline не падает).
        Результаты кэшируются по hash(text) для избежания повторных переводов.
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
                "Traduction échouée (%s) — texte original conservé.",
                self.MODEL_NAME,
                exc_info=True,
            )
            return text

    def translate_batch(self, texts: List[str]) -> List[str]:
        """Traduit chaque chaîne indépendamment, sans lever d'exception.

        Переводит каждую строку независимо; ошибка одной строки не прерывает batch.
        """
        return [self.translate(t) for t in texts]


# ---------------------------------------------------------------------------
# Traducteur russe → français / Переводчик с русского на французский
# ---------------------------------------------------------------------------

class RuFrTranslator(_BaseTranslator):
    """Traduit le russe en français via Helsinki-NLP/opus-mt-ru-fr.

    Переводит с русского на французский через Helsinki-NLP/opus-mt-ru-fr.
    Используется для отображения русских PDF-чанков на французском языке.
    """

    MODEL_NAME = "Helsinki-NLP/opus-mt-ru-fr"
