"""
models/base.py
──────────────
Tüm modellerin (ML + DL) ortak arayüzü.

ML modelleri TF-IDF tarafından üretilmiş 2D yoğun/seyrek matris ile fit olur.
DL modelleri token dizilerinden üretilmiş 3D-uyumlu integer matris ile fit
olur (Embedding ile genişletilir).

İki tip aynı arayüzü paylaşır:
    .fit(X_train, y_train)
    .predict(X_test) → np.ndarray
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    name: str = "base"

    @abstractmethod
    def fit(self, X_train, y_train) -> "BaseModel":
        ...

    @abstractmethod
    def predict(self, X_test) -> np.ndarray:
        ...
