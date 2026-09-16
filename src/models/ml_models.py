"""
models/ml_models.py
───────────────────
SVM, MNB, RF — TF-IDF girişiyle çalışan klasik makine öğrenmesi modelleri.

Her model BaseModel arayüzünü uygular: .fit(X, y) ve .predict(X).

Burada özellik (vocabulary) sabittir — vectorizer, runner tarafından
top_k(vocabulary) ile dışarıda kurulur, modele 2D matris gelir.
"""
from __future__ import annotations

import numpy as np

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier

from src.config import ML_CONFIG, RANDOM_SEED
from src.models.base import BaseModel


class _SklearnAdapter(BaseModel):
    """Hafif sklearn-sarmalayıcı."""

    def __init__(self, estimator) -> None:
        self.estimator = estimator

    def fit(self, X_train, y_train) -> "_SklearnAdapter":
        # RF yoğun matris bekler; SVM/MNB seyrekle de çalışır.
        if isinstance(self.estimator, RandomForestClassifier) and hasattr(X_train, "toarray"):
            X_train = X_train.toarray()
        self.estimator.fit(X_train, y_train)
        return self

    def predict(self, X_test) -> np.ndarray:
        if isinstance(self.estimator, RandomForestClassifier) and hasattr(X_test, "toarray"):
            X_test = X_test.toarray()
        return self.estimator.predict(X_test)


class SVMModel(_SklearnAdapter):
    name = "SVM"

    def __init__(self) -> None:
        super().__init__(
            SVC(
                kernel=ML_CONFIG.svm_kernel,
                C=ML_CONFIG.svm_C,
                random_state=RANDOM_SEED,
            )
        )


class MNBModel(_SklearnAdapter):
    name = "MNB"

    def __init__(self) -> None:
        super().__init__(MultinomialNB(alpha=ML_CONFIG.mnb_alpha))


class RFModel(_SklearnAdapter):
    name = "RF"

    def __init__(self) -> None:
        super().__init__(
            RandomForestClassifier(
                n_estimators=ML_CONFIG.rf_n_estimators,
                random_state=RANDOM_SEED,
                n_jobs=ML_CONFIG.rf_n_jobs,
            )
        )
