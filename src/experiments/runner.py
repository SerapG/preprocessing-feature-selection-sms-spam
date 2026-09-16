"""
experiments/runner.py
─────────────────────
Tüm deney grid'ini çalıştıran orkestratör.

Grid:
    languages × pipelines × methods × vocab_sizes × algorithms
        2     ×     4     ×    2    ×      6      ×      5     = 480 deney

Yüksek seviyeli akış:
    1. Dil için ham veriyi yükle.
    2. Stratified train/test split (yalnızca bir kez, seed=42).
    3. Her pipeline için preprocessing'i train ve test setine uygula.
    4. EĞİTİM tokenları üzerinden Gini ve DFS skorlarını hesapla
       (test seti öznitelik seçimine girmez; veri sızıntısı önlenir).
    5. Her (method, vocab_size) için:
         - ML: TfidfVectorizer(vocabulary=topK) → SVM/MNB/RF
         - DL: SequenceVectorizer(vocabulary=topK) → TextCNN/LSTM
    6. Her sonucu pandas DataFrame'e biriktir.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tqdm.auto import tqdm

from src.config import (
    ALGORITHMS,
    FEATURE_SELECTION_METHODS,
    LANGUAGES,
    ML_CONFIG,
    PIPELINES,
    RANDOM_SEED,
    TEST_SIZE,
    VOCAB_SIZES,
)
from src.data import EnglishLoader, Preprocessor, TurkishLoader
from src.feature_selection import DFSSelector, GiniSelector
from src.models import LSTMModel, MNBModel, RFModel, SVMModel, TextCNNModel


def macro_f1(y_true, y_pred) -> float:
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


@dataclass
class GridConfig:
    """Çalıştırılacak alt-grid'i tarif eder. `None` = full grid."""
    languages: tuple = LANGUAGES
    pipelines: tuple = PIPELINES
    methods: tuple = FEATURE_SELECTION_METHODS
    vocab_sizes: tuple = VOCAB_SIZES
    algorithms: tuple = ALGORITHMS

    @classmethod
    def smoke(cls) -> "GridConfig":
        """Hızlı geliştirme/sanity-check için minimal grid."""
        return cls(
            languages=("english",),
            pipelines=(("basic", False, False),),
            methods=("gini",),
            vocab_sizes=(100,),
            algorithms=ALGORITHMS,
        )


class ExperimentRunner:
    """
    Grid'i çalıştırır, sonuçları long-format DataFrame olarak döner.

    Sütunlar: pipeline | language | method | vocab_size | algorithm | macro_f1
    """

    def __init__(self, grid: GridConfig | None = None) -> None:
        self.grid = grid or GridConfig()
        self._results: list[dict] = []

    # ── Loader fabrika ────────────────────────────────────────────────────
    @staticmethod
    def _load_raw(language: str) -> pd.DataFrame:
        if language == "english":
            return EnglishLoader().load()
        if language == "turkish":
            return TurkishLoader().load()
        raise ValueError(f"Unknown language: {language!r}")

    @staticmethod
    def _build_selector(method: str):
        if method == "gini":
            return GiniSelector()
        if method == "dfs":
            return DFSSelector()
        raise ValueError(f"Unknown method: {method!r}")

    # ── Tek (lang, pipeline) için tüm method × vocab × model alt-grid'i ──

    def _run_for_language_pipeline(
        self,
        language: str,
        pipeline_id: str,
        rm_stop: bool,
        ap_stem: bool,
        texts_train: list[str],
        texts_test: list[str],
        y_train: list[str],
        y_test: list[str],
        outer_bar: tqdm,
    ) -> None:
        # 1) Preprocessing
        prep = Preprocessor(
            language=language,
            remove_stopwords=rm_stop,
            apply_stemming=ap_stem,
        )
        train_tokens = prep.transform_many(texts_train)
        test_tokens = prep.transform_many(texts_test)

        # ML için TF-IDF input metni: token listesini boşlukla join
        train_texts_proc = [" ".join(toks) for toks in train_tokens]
        test_texts_proc = [" ".join(toks) for toks in test_tokens]

        # 2) Feature selectors (TRAIN'den hesaplanır)
        selectors = {}
        for m in self.grid.methods:
            sel = self._build_selector(m)
            sel.fit(train_tokens, y_train)
            selectors[m] = sel

        # 3) Method × vocab_size × algoritma döngüsü
        for method in self.grid.methods:
            selector = selectors[method]
            for k in self.grid.vocab_sizes:
                vocabulary = selector.top_k(k)

                # — ML: TF-IDF bir kez kurulur, üç ML modeli paylaşır —
                if any(a in self.grid.algorithms for a in ("SVM", "MNB", "RF")):
                    if not vocabulary:
                        # Çok küçük vocab veya tüm skorlar 0 ise atla
                        continue
                    vec = TfidfVectorizer(
                        vocabulary=vocabulary,
                        sublinear_tf=ML_CONFIG.tfidf_sublinear,
                    )
                    X_train = vec.fit_transform(train_texts_proc)
                    X_test = vec.transform(test_texts_proc)

                    if "SVM" in self.grid.algorithms:
                        self._run_one(
                            SVMModel(), X_train, y_train, X_test, y_test,
                            language, pipeline_id, method, k, "SVM", outer_bar,
                        )
                    if "MNB" in self.grid.algorithms:
                        self._run_one(
                            MNBModel(), X_train, y_train, X_test, y_test,
                            language, pipeline_id, method, k, "MNB", outer_bar,
                        )
                    if "RF" in self.grid.algorithms:
                        self._run_one(
                            RFModel(), X_train, y_train, X_test, y_test,
                            language, pipeline_id, method, k, "RF", outer_bar,
                        )

                # — DL: train_tokens / test_tokens ham token listeleri —
                if "TextCNN" in self.grid.algorithms:
                    self._run_one(
                        TextCNNModel(vocabulary),
                        train_tokens, y_train, test_tokens, y_test,
                        language, pipeline_id, method, k, "TextCNN", outer_bar,
                    )
                if "LSTM" in self.grid.algorithms:
                    self._run_one(
                        LSTMModel(vocabulary),
                        train_tokens, y_train, test_tokens, y_test,
                        language, pipeline_id, method, k, "LSTM", outer_bar,
                    )

    def _run_one(
        self,
        model,
        X_train, y_train, X_test, y_test,
        language: str, pipeline_id: str, method: str,
        vocab_size: int, algorithm: str,
        bar: tqdm,
    ) -> None:
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            f1 = macro_f1(y_test, y_pred)
        except Exception as exc:  # noqa: BLE001 — log et, devam et
            f1 = float("nan")
            tqdm.write(
                f"  ⚠ {language}/{pipeline_id}/{method}/VS={vocab_size}/{algorithm}: {exc}"
            )

        self._results.append(
            {
                "pipeline": pipeline_id,
                "language": language,
                "method": method,
                "vocab_size": vocab_size,
                "algorithm": algorithm,
                "macro_f1": round(f1, 4),
            }
        )
        bar.update(1)
        bar.set_postfix_str(
            f"{language}|{pipeline_id}|{method}|VS={vocab_size}|{algorithm}={f1:.4f}"
        )

    # ── Ana giriş ─────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        # Toplam deney sayısı (progress bar boyutu)
        n_total = (
            len(self.grid.languages)
            * len(self.grid.pipelines)
            * len(self.grid.methods)
            * len(self.grid.vocab_sizes)
            * len(self.grid.algorithms)
        )

        bar = tqdm(total=n_total, desc="Experiments", ncols=100)
        try:
            for language in self.grid.languages:
                df = self._load_raw(language)
                # Stratified split — bir kez per language
                idx_train, idx_test = train_test_split(
                    np.arange(len(df)),
                    test_size=TEST_SIZE,
                    random_state=RANDOM_SEED,
                    stratify=df["label"].values,
                )
                texts_all = df["text"].tolist()
                labels_all = df["label"].tolist()
                texts_train = [texts_all[i] for i in idx_train]
                texts_test = [texts_all[i] for i in idx_test]
                y_train = [labels_all[i] for i in idx_train]
                y_test = [labels_all[i] for i in idx_test]

                tqdm.write(
                    f"\n[{language}] toplam={len(df)} | train={len(y_train)} | test={len(y_test)}"
                )

                for pipeline_id, rm_stop, ap_stem in self.grid.pipelines:
                    self._run_for_language_pipeline(
                        language, pipeline_id, rm_stop, ap_stem,
                        texts_train, texts_test, y_train, y_test,
                        outer_bar=bar,
                    )
        finally:
            bar.close()

        return pd.DataFrame(self._results)
