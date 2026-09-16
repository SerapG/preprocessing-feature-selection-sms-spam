"""
experiments/reporter.py
───────────────────────
Long-format sonuç DataFrame'ini disk'e yazar ve makaleye doğrudan
kopyalanabilir 16 pivot tablo üretir (4 pipeline × 2 dil × 2 yöntem).

Her pivot tablo: satır = VocabSize, sütun = Algoritma, değer = Macro-F1.
"""
from __future__ import annotations

import glob
from pathlib import Path

import pandas as pd

from src.config import ALGORITHMS, RESULTS_DIR, VOCAB_SIZES


def next_run_number(base_dir: Path = RESULTS_DIR) -> int:
    base_dir.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in base_dir.iterdir():
        if p.is_dir() and p.name.isdigit():
            nums.append(int(p.name))
    return (max(nums) + 1) if nums else 1


class Reporter:
    """
    Sonuç DataFrame'ini diske yazar ve pivot tablolar üretir.

    Çıktı klasörü: results/<run>/
        ├── final_results.csv
        └── tables/
            ├── english_basic_gini.csv
            ├── english_basic_dfs.csv
            ├── ... (16 dosya)
            └── turkish_full_dfs.csv
    """

    def __init__(self, run_no: int | None = None, base_dir: Path = RESULTS_DIR) -> None:
        self.run_no = run_no if run_no is not None else next_run_number(base_dir)
        self.run_dir = base_dir / str(self.run_no)
        self.tables_dir = self.run_dir / "tables"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    def save(self, results: pd.DataFrame) -> None:
        # Long-format sonuçlar
        long_path = self.run_dir / "final_results.csv"
        results.to_csv(long_path, index=False, encoding="utf-8-sig")
        print(f"\n✓ Long-format sonuçlar    : {long_path}")

        # Pivot tabloları
        self._write_pivots(results)

        # Konsol özeti
        self._print_console_summary(results)

    def _write_pivots(self, results: pd.DataFrame) -> None:
        # VocabSize'ı azalan sırada (500, 300, 100, ...) tut.
        vs_order = [v for v in VOCAB_SIZES if v in results["vocab_size"].unique()]
        algo_order = [a for a in ALGORITHMS if a in results["algorithm"].unique()]

        groups = results.groupby(["language", "pipeline", "method"])
        for (lang, pipe, method), sub in groups:
            pivot = (
                sub.pivot_table(
                    index="vocab_size",
                    columns="algorithm",
                    values="macro_f1",
                    aggfunc="first",
                )
                .reindex(index=vs_order, columns=algo_order)
            )
            pivot.index.name = "VS"
            fname = f"{lang}_{pipe}_{method}.csv"
            out = self.tables_dir / fname
            pivot.to_csv(out, encoding="utf-8-sig", float_format="%.4f")
            print(f"  → {out.relative_to(self.run_dir.parent)}")

    def _print_console_summary(self, results: pd.DataFrame) -> None:
        print("\n" + "=" * 80)
        print("  ÖZET — Macro-F1 (her pipeline × dil × yöntem için pivot)")
        print("=" * 80)

        vs_order = [v for v in VOCAB_SIZES if v in results["vocab_size"].unique()]
        algo_order = [a for a in ALGORITHMS if a in results["algorithm"].unique()]

        groups = results.groupby(["language", "pipeline", "method"])
        for (lang, pipe, method), sub in groups:
            pivot = (
                sub.pivot_table(
                    index="vocab_size",
                    columns="algorithm",
                    values="macro_f1",
                    aggfunc="first",
                )
                .reindex(index=vs_order, columns=algo_order)
            )
            pivot.index.name = "VS"
            print(f"\n[{lang.upper()}]  Pipeline={pipe}  |  Method={method}")
            print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
