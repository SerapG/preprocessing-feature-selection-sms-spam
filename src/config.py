"""
config.py
─────────
Tüm sabitler, hiperparametreler ve yapılandırma parametreleri burada toplanır.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


# ─── PROJE KÖK YOLU ──────────────────────────────────────────────────────────
# Proje çalışma dizinine göre tüm dataset yolları çözümlenir.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Çıktıların kaydedileceği klasör.
RESULTS_DIR = PROJECT_ROOT / "results"


# ─── VERİ KAYNAKLARI (otomatik bulma) ────────────────────────────────────────
# English_sms_spam/ ve TurkishSMS/ klasörlerini bulana kadar bu dosyanın
# bulunduğu dizinden yukarı doğru tarar; veri setleri repo köküne
# konduğunda ek yol ayarı gerekmez.

def _find_dataset_root() -> Path:
    """English_sms_spam ve TurkishSMS klasörlerini içeren ata-dizini bul."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "English_sms_spam").is_dir() and (parent / "TurkishSMS").is_dir():
            return parent
    # Geri düşüş: cwd'den dene.
    cwd = Path.cwd()
    if (cwd / "English_sms_spam").is_dir() and (cwd / "TurkishSMS").is_dir():
        return cwd
    raise FileNotFoundError(
        "Veri kök dizini bulunamadı. 'English_sms_spam/' ve 'TurkishSMS/' "
        "klasörlerini içeren bir dizinden çalıştırın ya da PROJECT_ROOT'u elle ayarlayın."
    )


_DATA_ROOT = _find_dataset_root()

ENGLISH_DATA_PATH = _DATA_ROOT / "English_sms_spam" / "SMSSpamCollection"
TURKISH_HAM_PATH = _DATA_ROOT / "TurkishSMS" / "sms_legitimate.txt"
TURKISH_SPAM_PATH = _DATA_ROOT / "TurkishSMS" / "sms_spam.txt"

# Türkçe için flat .txt dosyaları kanoniktir; C01/C02 klasörleri kullanılmaz
# (içerik aynı, dosya adları yanıltıcı: tüm dosyalar *.ham.txt uzantılı).


# ─── DENEY GRID'İ ────────────────────────────────────────────────────────────

LANGUAGES = ("english", "turkish")

PIPELINES = (
    # (pipeline_id, remove_stopwords, apply_stemming)
    ("basic",    False, False),
    ("stopword", True,  False),
    ("stemming", False, True),
    ("full",     True,  True),
)

FEATURE_SELECTION_METHODS = ("gini", "dfs")

VOCAB_SIZES = (500, 300, 100, 50, 30, 10)

ALGORITHMS = ("SVM", "MNB", "RF", "TextCNN", "LSTM")


# ─── REPRODUCIBILITY ─────────────────────────────────────────────────────────
RANDOM_SEED = 42
TEST_SIZE = 0.30


# ─── ML HİPERPARAMETRELERİ ───────────────────────────────────────────────────

@dataclass(frozen=True)
class MLConfig:
    svm_kernel: str = "linear"
    svm_C: float = 1.0
    mnb_alpha: float = 1.0
    rf_n_estimators: int = 200
    rf_n_jobs: int = -1
    tfidf_sublinear: bool = True

ML_CONFIG = MLConfig()


# ─── DL HİPERPARAMETRELERİ ───────────────────────────────────────────────────

@dataclass(frozen=True)
class DLConfig:
    embed_dim: int = 64
    max_len: int = 80
    batch_size: int = 32
    epochs: int = 15
    learning_rate: float = 1e-3
    dropout: float = 0.5
    early_stop_patience: int = 3

    # TextCNN
    cnn_kernel_sizes: tuple = (3, 4, 5)
    cnn_num_filters: int = 100

    # LSTM
    lstm_hidden_size: int = 128
    lstm_num_layers: int = 1

DL_CONFIG = DLConfig()


# ─── SMALL NUMERICAL CONSTANT ────────────────────────────────────────────────
EPSILON = 1e-10
