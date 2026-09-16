"""
models/lstm.py
──────────────
LSTM tabanlı sınıflandırıcı.

Mimari:
    Embedding(padding_idx=0)
    → LSTM (1 katman, hidden=128, batch_first=True)
    → son hidden state → dropout → FC → 2-class logits

Girdi: SequenceVectorizer ile elde edilen [N, MAX_LEN] integer tensor.

NOT — Ortak bir <OOV> indeksi kullanılmaz: vocabulary'de olmayan tokenlar
dizilerden silinir. Küçük VS değerlerinde ortak bir OOV indeksi dizileri
birbirine benzetip ayırt ediciliği azaltacağı için bu tercih yapılmıştır.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from src.config import DL_CONFIG, RANDOM_SEED
from src.models.base import BaseModel
from src.models.sequence import (
    SequenceVectorizer,
    class_weights,
    make_loader,
    set_torch_seeds,
)


class _LSTMNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Gerçek (non-PAD) uzunluk hesabı — packed sequence için.
        # Filtre sonrası bazı örnekler tamamen PAD olabilir (uzunluk 0);
        # pack_padded_sequence sıfır uzunluğu kabul etmez → minimum 1'e zorla.
        lengths = (x != 0).sum(dim=1).clamp(min=1).cpu()
        emb = self.embedding(x)                            # [B, L, D]
        packed = nn.utils.rnn.pack_padded_sequence(
            emb, lengths, batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last = h_n[-1]                                     # [B, H]
        last = self.dropout(last)
        return self.fc(last)                               # [B, num_classes]


class LSTMModel(BaseModel):
    name = "LSTM"

    def __init__(self, vocabulary: list[str]) -> None:
        set_torch_seeds(RANDOM_SEED)
        self.vectorizer = SequenceVectorizer(vocabulary)
        self.cfg = DL_CONFIG
        self.device = torch.device("cpu")
        self.net: _LSTMNet | None = None
        self._classes: list = []

    def _build_net(self) -> _LSTMNet:
        return _LSTMNet(
            vocab_size=self.vectorizer.vocab_size,
            embed_dim=self.cfg.embed_dim,
            hidden_size=self.cfg.lstm_hidden_size,
            num_layers=self.cfg.lstm_num_layers,
            dropout=self.cfg.dropout,
            num_classes=2,
        )

    def fit(self, train_tokens: list[list[str]], y_train: list[str]) -> "LSTMModel":
        self._classes = sorted(set(y_train))
        cls2idx = {c: i for i, c in enumerate(self._classes)}
        y_idx = np.array([cls2idx[c] for c in y_train], dtype=np.int64)

        X = self.vectorizer.transform(train_tokens).to(self.device)
        y = torch.from_numpy(y_idx).to(self.device)

        self.net = self._build_net().to(self.device)
        optimizer = torch.optim.Adam(
            self.net.parameters(),
            lr=self.cfg.learning_rate,
        )
        # Sınıf dengesizliği için ağırlıklı kayıp — LSTM'in çoğunluk
        # sınıfına çökmesini engeller.
        weights = class_weights(y_train, self._classes).to(self.device)
        criterion = nn.CrossEntropyLoss(weight=weights)
        loader = make_loader(X, y, batch_size=self.cfg.batch_size, shuffle=True)

        best_loss = float("inf")
        bad_epochs = 0
        for _ in range(self.cfg.epochs):
            self.net.train()
            total = 0.0
            for xb, yb in loader:
                optimizer.zero_grad()
                logits = self.net(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total += loss.item() * xb.size(0)
            avg = total / len(X)
            if avg < best_loss - 1e-4:
                best_loss = avg
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= self.cfg.early_stop_patience:
                    break
        return self

    def predict(self, test_tokens: list[list[str]]) -> np.ndarray:
        if self.net is None:
            raise RuntimeError("LSTMModel not fitted.")
        X = self.vectorizer.transform(test_tokens).to(self.device)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(X)
            idx = logits.argmax(dim=-1).cpu().numpy()
        return np.array([self._classes[i] for i in idx])
