"""
feature_selection/base.py
─────────────────────────
Tüm öznitelik seçim yöntemlerinin ortak arayüzü ve istatistik tablosu.

İstatistik tablosu (binary occurrence) bir kez hesaplanır, hem Gini hem DFS
aynı tabloyu kullanır → yeniden hesaplama maliyeti yok.

Skorlar yalnızca eğitim setinden hesaplanır.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Iterable

from src.config import EPSILON


def build_stats(
    train_tokens: list[list[str]],
    labels: Iterable[str],
) -> dict:
    """
    Eğitim seti üzerinde her kelime × sınıf için belge frekansını hesaplar.
    Bir belgede kelime varsa 1 sayar (binary occurrence).

    Returns
    -------
    dict
        classes      : list[str]     → benzersiz sınıflar
        N            : int           → toplam belge sayısı
        class_counts : dict          → sınıf → belge sayısı
        term_class   : dict          → kelime → sınıf → belge sayısı
        doc_counts   : dict          → kelime → toplam belge sayısı
    """
    labels = list(labels)
    if len(train_tokens) != len(labels):
        raise ValueError("train_tokens and labels length mismatch.")

    N = len(train_tokens)
    class_counts: dict[str, int] = defaultdict(int)
    term_class: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    doc_counts: dict[str, int] = defaultdict(int)

    for tokens, label in zip(train_tokens, labels):
        class_counts[label] += 1
        seen: set[str] = set()
        for tok in tokens:
            if tok in seen:
                continue
            seen.add(tok)
            term_class[tok][label] += 1
            doc_counts[tok] += 1

    return {
        "classes": sorted(class_counts.keys()),
        "N": N,
        "class_counts": dict(class_counts),
        "term_class": {k: dict(v) for k, v in term_class.items()},
        "doc_counts": dict(doc_counts),
    }


class BaseFeatureSelector(ABC):
    """
    fit() → tüm vocabulary için skor matrisini hesapla.
    top_k(k) → ilk k kelimeyi (skora göre azalan) döndür.
    """

    name: str = "base"

    def __init__(self) -> None:
        self._scores: dict[str, float] = {}
        self._ranked: list[str] = []
        self._stats: dict | None = None

    def fit(self, train_tokens: list[list[str]], labels: Iterable[str]) -> "BaseFeatureSelector":
        self._stats = build_stats(train_tokens, labels)
        self._scores = {
            term: self._score(term, self._stats)
            for term in self._stats["doc_counts"]
        }
        self._ranked = sorted(self._scores, key=self._scores.get, reverse=True)
        return self

    def top_k(self, k: int) -> list[str]:
        if not self._ranked:
            raise RuntimeError(f"{self.name}.fit(...) must be called before top_k().")
        return self._ranked[:k]

    @property
    def scores(self) -> dict[str, float]:
        return dict(self._scores)

    @abstractmethod
    def _score(self, term: str, stats: dict) -> float:
        """Verilen kelimenin skorunu hesapla (alt sınıf override eder)."""
        ...

    @staticmethod
    def _eps() -> float:
        return EPSILON
