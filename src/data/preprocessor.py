"""
data/preprocessor.py
────────────────────
Metin ön işleme: 4 pipeline aynı sınıf parametre kombinasyonu ile elde edilir.

Ön işleme pipeline'ları:
    P1 basic    : lower + punct + tokenize
    P2 stopword : P1 + stop-word removal
    P3 stemming : P1 + stemming
    P4 full     : P1 + stop-word removal + stemming

Türkçe için lowercasing kuralı:
    İ → i, I → ı, sonra .lower()
"""
from __future__ import annotations

import re
import string
import warnings
from typing import Callable

# NLTK stop-words (gerekirse indirilir).
try:
    import nltk
    from nltk.corpus import stopwords as _nltk_stopwords

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)
    _NLTK_OK = True
except ImportError:
    _NLTK_OK = False
    warnings.warn("nltk not installed; stop-word removal disabled.")

# Snowball stemmer (English & Turkish).
try:
    import snowballstemmer

    _SNOWBALL_OK = True
except ImportError:
    _SNOWBALL_OK = False
    warnings.warn("snowballstemmer not installed; stemming disabled.")


_PUNCT_PATTERN = re.compile(r"[" + re.escape(string.punctuation) + r"\d]+")


def _turkish_lower(text: str) -> str:
    """
    Türkçe için doğru küçük harfe çevirme.

    Standart .lower() bazı sistemlerde 'I' → 'i' (yanlış) verir.
    Türkçe'de doğrusu 'I' → 'ı', 'İ' → 'i' olmalıdır.
    """
    return text.replace("İ", "i").replace("I", "ı").lower()


class Preprocessor:
    """
    Tek bir Preprocessor örneği = bir pipeline.

    Parameters
    ----------
    language : 'english' | 'turkish'
    remove_stopwords : bool
    apply_stemming : bool
    """

    def __init__(
        self,
        language: str,
        remove_stopwords: bool = False,
        apply_stemming: bool = False,
    ) -> None:
        if language not in ("english", "turkish"):
            raise ValueError(f"Unknown language: {language!r}")
        self.language = language
        self.remove_stopwords = remove_stopwords
        self.apply_stemming = apply_stemming

        self._lower: Callable[[str], str] = (
            _turkish_lower if language == "turkish" else str.lower
        )

        self._stopwords: set[str] = self._build_stopwords()
        self._stem: Callable[[str], str] = self._build_stemmer()

    # ── Yardımcı kurucular ────────────────────────────────────────────────

    def _build_stopwords(self) -> set[str]:
        if not self.remove_stopwords or not _NLTK_OK:
            return set()
        words = set(_nltk_stopwords.words(self.language))
        if self.language == "turkish":
            # Türkçe stopword'leri de Türkçe lowercaser'dan geçir
            # (NLTK normalde 'İ' içerebilir).
            words = {_turkish_lower(w) for w in words}
        return words

    def _build_stemmer(self) -> Callable[[str], str]:
        if not self.apply_stemming or not _SNOWBALL_OK:
            return lambda w: w  # identity
        algo = "english" if self.language == "english" else "turkish"
        stemmer = snowballstemmer.stemmer(algo)
        return lambda w: stemmer.stemWord(w)

    # ── Ana API ───────────────────────────────────────────────────────────

    def transform(self, text: str) -> list[str]:
        """Tek bir SMS metnini token listesine dönüştür."""
        text = self._lower(str(text))
        # Noktalama + rakam → boşluk (parçalamak yerine ayırarak token üret)
        text = _PUNCT_PATTERN.sub(" ", text)
        tokens = text.split()
        if self.remove_stopwords and self._stopwords:
            tokens = [t for t in tokens if t not in self._stopwords]
        if self.apply_stemming:
            tokens = [self._stem(t) for t in tokens]
        # Boş string'leri at (stemmer bazen "" döndürebilir)
        return [t for t in tokens if t]

    def transform_many(self, texts: list[str]) -> list[list[str]]:
        return [self.transform(t) for t in texts]
