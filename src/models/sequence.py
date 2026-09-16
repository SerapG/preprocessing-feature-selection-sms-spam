"""
models/sequence.py
──────────────────
DL modelleri için TOKEN → INTEGER SEQUENCE dönüştürücü ve eğitim/yardımcılar.

Kurallar:
    - Vocabulary bir FİLTRE: top-K'da olmayan tokenlar diziden SİLİNİR;
      <OOV> indeksi YOKTUR.
    - Filtreden geçen tokenlar metindeki ORİJİNAL SIRADA tutulur.

İndeksleme:
    0       → <PAD>  (padding_idx olarak Embedding'e verilir)
    1..K    → vocabulary kelimeleri

Maksimum uzunluk MAX_LEN'i geçen diziler kesilir; daha kısa olanlar PAD ile
sağdan tamamlanır.
"""
from __future__ import annotations

import os
import random
from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.config import DL_CONFIG, RANDOM_SEED


PAD_INDEX = 0


def class_weights(y: list[str], classes: list[str]) -> torch.Tensor:
    """
    Sınıf dengesizliği için CrossEntropyLoss'a verilecek ağırlıklar.
    English SMS Spam aşırı dengesiz (~87% ham); ağırlık olmadan LSTM
    çoğunluk sınıfına çöker → makro-F1 çoğunluk sınıfı F1'inin yarısı.

    Inverse-frequency: w_c = N / (n_classes * n_c)
    """
    counts = Counter(y)
    n = len(y)
    k = len(classes)
    weights = [n / (k * counts[c]) for c in classes]
    return torch.tensor(weights, dtype=torch.float32)


def set_torch_seeds(seed: int = RANDOM_SEED) -> None:
    """Tekrarlanabilirlik: numpy + torch + python random seed'lerini sabitle."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


class SequenceVectorizer:
    """
    Token listelerini [N, MAX_LEN] integer matrisine çevirir.

    Strict filter:
    - Vocabulary'de OLMAYAN token: atılır (sıraya etkisi yok).
    - Vocabulary'deki token: kendi indeksine (1..K) çevrilir.
    - Eksik yerler PAD_INDEX (0) ile sağdan doldurulur.
    """

    def __init__(self, vocabulary: list[str], max_len: int = DL_CONFIG.max_len) -> None:
        self.vocabulary = list(vocabulary)
        self.max_len = max_len
        # +1: PAD için 0'ı rezerve et.
        self._word2idx = {w: i + 1 for i, w in enumerate(self.vocabulary)}
        self.vocab_size = len(self.vocabulary) + 1  # PAD dahil

    def transform(self, token_lists: list[list[str]]) -> torch.Tensor:
        n = len(token_lists)
        out = np.full((n, self.max_len), PAD_INDEX, dtype=np.int64)
        for i, tokens in enumerate(token_lists):
            # Strict filter — sıra korunur, OOV silinir.
            ids = [self._word2idx[t] for t in tokens if t in self._word2idx]
            ids = ids[: self.max_len]
            out[i, : len(ids)] = ids
        return torch.from_numpy(out)


def make_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = DL_CONFIG.batch_size,
    shuffle: bool = True,
) -> DataLoader:
    """Reproducible DataLoader (shuffle için generator seed'lenir)."""
    g = torch.Generator()
    g.manual_seed(RANDOM_SEED)
    return DataLoader(
        TensorDataset(X, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=g,
    )
