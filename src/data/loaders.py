"""
data/loaders.py
───────────────
SMS veri kümelerini diskten okuyan sınıflar.

Her loader, pd.DataFrame döndürür:
    'label' : 'ham' | 'spam'
    'text'  : ham metin (preprocessing UYGULANMAMIŞ)

Preprocessing'i loaderdan ayırmamızın nedeni: aynı veri seti üzerinde
4 farklı pipeline'ı tekrar tekrar çalıştırırken disk I/O'yu birden fazla
yapmamak (loader bir kez çağrılır, preprocessing 4 kez uygulanır).

"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pandas as pd

from src.config import (
    ENGLISH_DATA_PATH,
    TURKISH_HAM_PATH,
    TURKISH_SPAM_PATH,
)


# Türkçe dosyaları için denenecek encoding sırası.
_TURKISH_ENCODINGS = ("utf-8", "utf-8-sig", "cp1254", "iso-8859-9", "latin-1")


class BaseLoader(ABC):
    """Tüm loader'ların ortak arayüzü."""

    @abstractmethod
    def load(self) -> pd.DataFrame:
        """
        Returns
        -------
        pd.DataFrame
            'label' ('ham'/'spam') ve 'text' (str) sütunlarına sahip.
        """
        ...


# ─── İNGİLİZCE ────────────────────────────────────────────────────────────────

class EnglishLoader(BaseLoader):
    """UCI SMS Spam Collection (tab-ayrımlı, başlıksız)."""

    def __init__(self, path: str | Path = ENGLISH_DATA_PATH) -> None:
        self.path = Path(path)

    def load(self) -> pd.DataFrame:
        df = pd.read_csv(
            self.path,
            sep="\t",
            header=None,
            names=["label", "text"],
            encoding="utf-8",
            on_bad_lines="skip",
        )
        df["label"] = df["label"].str.strip().str.lower()
        df["text"] = df["text"].astype(str)
        df = df[df["label"].isin(("ham", "spam"))].reset_index(drop=True)
        return df


# ─── TÜRKÇE ───────────────────────────────────────────────────────────────────

class TurkishLoader(BaseLoader):
    """
    TurkishSMS (Uysal et al., 2012) — flat .txt dosyaları.

    Her satır = 1 SMS. Etiket dosya isminden geliyor (legitimate/spam).
    """

    def __init__(
        self,
        ham_path: str | Path = TURKISH_HAM_PATH,
        spam_path: str | Path = TURKISH_SPAM_PATH,
    ) -> None:
        self.ham_path = Path(ham_path)
        self.spam_path = Path(spam_path)

    def load(self) -> pd.DataFrame:
        ham_messages = self._read_lines(self.ham_path)
        spam_messages = self._read_lines(self.spam_path)

        records = (
            [{"label": "ham", "text": m} for m in ham_messages]
            + [{"label": "spam", "text": m} for m in spam_messages]
        )
        return pd.DataFrame(records)

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        """Encoding fall-back ile satır satır oku, boş satırları at."""
        text: str | None = None
        for enc in _TURKISH_ENCODINGS:
            try:
                with open(path, "r", encoding=enc) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        if text is None:  # son çare
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        # BOM temizle, satırları sıyır, boşları at.
        text = text.lstrip("\ufeff")
        return [ln.strip() for ln in text.splitlines() if ln.strip()]
