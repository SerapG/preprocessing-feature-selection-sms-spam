"""
feature_selection/dfs.py
────────────────────────
Distinguishing Feature Selector (Uysal & Gunal, 2012):

    DFS(t) = Σⱼ P(Cⱼ | t) / ( P(t̄ | Cⱼ) + P(t | C̄ⱼ) + 1 )

    P(Cⱼ | t)   = n(t, Cⱼ) / n(t)
    P(t̄ | Cⱼ)  = 1 - n(t, Cⱼ) / n(Cⱼ)
    P(t | C̄ⱼ)  = (n(t) - n(t, Cⱼ)) / (N - n(Cⱼ))

Paydadaki +1 → skoru (0, 1) aralığında tutar ve sıfıra bölmeyi engeller.
"""
from __future__ import annotations

from src.feature_selection.base import BaseFeatureSelector


class DFSSelector(BaseFeatureSelector):
    name = "dfs"

    def _score(self, term: str, stats: dict) -> float:
        N = stats["N"]
        n_t = stats["doc_counts"].get(term, 0)
        score = 0.0
        for cls in stats["classes"]:
            n_c = stats["class_counts"].get(cls, 0)
            n_t_c = stats["term_class"].get(term, {}).get(cls, 0)

            p_c_given_t = n_t_c / (n_t + self._eps())
            p_not_t_given_c = 1.0 - n_t_c / (n_c + self._eps())
            p_t_given_not_c = (n_t - n_t_c) / ((N - n_c) + self._eps())

            score += p_c_given_t / (p_not_t_given_c + p_t_given_not_c + 1.0)
        return score
