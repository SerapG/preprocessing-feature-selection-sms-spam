"""
models/textcnn.py
─────────────────
TextCNN (Kim, 2014) — çoklu kernel-boyutlu 1D konvolüsyonlar.

Mimari:
    Embedding(padding_idx=0)
    → [Conv1d(k=3) | Conv1d(k=4) | Conv1d(k=5)]  (her biri num_filters)
    → ReLU → max-pool over time
    → concat → dropout → FC → 2-class logits

Girdi: SequenceVectorizer ile elde edilen [N, MAX_LEN] integer tensor.

Bu modele OOV token gelmez; padding her zaman index=0.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import DL_CONFIG, RANDOM_SEED
from src.models.base import BaseModel
from src.models.sequence import (
    SequenceVectorizer,
    class_weights,
    make_loader,
    set_torch_seeds,
)


class _TextCNNNet(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        kernel_sizes: tuple[int, ...],
        num_filters: int,
        dropout: float,
        num_classes: int = 2,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embed_dim,
            padding_idx=0,
        )
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embed_dim,
                    out_channels=num_filters,
                    kernel_size=k,
                )
                for k in kernel_sizes
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L]
        emb = self.embedding(x)              # [B, L, D]
        emb = emb.transpose(1, 2)            # [B, D, L]
        feats = []
        for conv in self.convs:
            h = F.relu(conv(emb))            # [B, F, L-k+1]
            # Eğer dizi kernel'dan kısaysa max boyutu 1'e zorlamak için kontrol
            if h.size(-1) <= 0:
                # max_len < min(kernel_sizes) durumu — pratik değil ama güvenlik
                h = torch.zeros(h.size(0), h.size(1), 1, device=x.device)
            h = F.max_pool1d(h, kernel_size=h.size(-1)).squeeze(-1)  # [B, F]
            feats.append(h)
        out = torch.cat(feats, dim=1)        # [B, F * len(kernel_sizes)]
        out = self.dropout(out)
        return self.fc(out)                  # [B, num_classes]


class TextCNNModel(BaseModel):
    name = "TextCNN"

    def __init__(self, vocabulary: list[str]) -> None:
        set_torch_seeds(RANDOM_SEED)
        self.vectorizer = SequenceVectorizer(vocabulary)
        self.cfg = DL_CONFIG
        self.device = torch.device("cpu")  # AMD GPU yok, CPU sabit.
        self.net: _TextCNNNet | None = None
        self._classes: list = []

    def _build_net(self) -> _TextCNNNet:
        return _TextCNNNet(
            vocab_size=self.vectorizer.vocab_size,
            embed_dim=self.cfg.embed_dim,
            kernel_sizes=self.cfg.cnn_kernel_sizes,
            num_filters=self.cfg.cnn_num_filters,
            dropout=self.cfg.dropout,
            num_classes=2,
        )

    def fit(self, train_tokens: list[list[str]], y_train: list[str]) -> "TextCNNModel":
        # Kategori → integer (sklearn LabelEncoder gibi, ama bağımsız)
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
        # Sınıf dengesizliği için ağırlıklı kayıp.
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
            raise RuntimeError("TextCNNModel not fitted.")
        X = self.vectorizer.transform(test_tokens).to(self.device)
        self.net.eval()
        with torch.no_grad():
            logits = self.net(X)
            idx = logits.argmax(dim=-1).cpu().numpy()
        return np.array([self._classes[i] for i in idx])
