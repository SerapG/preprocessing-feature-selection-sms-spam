"""
feature_selection/gini.py
─────────────────────────
Gini İndeksi:

    GI(t) = Σⱼ P(t | Cⱼ)² · P(Cⱼ | t)²   (Shang et al., 2007)

    P(t | Cⱼ) = n(t, Cⱼ) / n(Cⱼ)     → Cⱼ'de t'yi içeren belge oranı
    P(Cⱼ | t) = n(t, Cⱼ) / n(t)      → t'yi içerenler arasında Cⱼ oranı

Yüksek skor → terim sınıf ayırt edicidir.
"""
from __future__ import annotations

from src.feature_selection.base import BaseFeatureSelector


class GiniSelector(BaseFeatureSelector):
    name = "gini"

    def _score(self, term: str, stats: dict) -> float:
        n_t = stats["doc_counts"].get(term, 0)
        score = 0.0
        for cls in stats["classes"]:
            n_c = stats["class_counts"].get(cls, 0)
            n_t_c = stats["term_class"].get(term, {}).get(cls, 0)

            p_t_given_c = n_t_c / (n_c + self._eps())
            p_c_given_t = n_t_c / (n_t + self._eps())
            score += (p_t_given_c ** 2) * (p_c_given_t ** 2)
        return score
