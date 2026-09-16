"""
run.py
──────
Tek satırlık başlatma noktası.

Kullanım:
    python run.py            # Tüm grid (~29 dk, CPU)
    python run.py --smoke    # Hızlı duman testi (~1 dk)
"""
from __future__ import annotations

import argparse
import sys
import time

# Konsol çıktısının Türkçe karakterleri Windows'ta doğru göstermesi için.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from src.experiments import ExperimentRunner, Reporter
from src.experiments.runner import GridConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="SMS spam ablation runner.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Sadece minimal grid (1 dil × 1 pipeline × 1 yöntem × 1 vs × 5 algo).",
    )
    args = parser.parse_args()

    grid = GridConfig.smoke() if args.smoke else GridConfig()
    mode = "SMOKE" if args.smoke else "FULL"

    print("=" * 80)
    print(f"  SMS Spam Ablation — Mode: {mode}")
    print("=" * 80)
    print(f"  Languages   : {grid.languages}")
    print(f"  Pipelines   : {[p[0] for p in grid.pipelines]}")
    print(f"  Methods     : {grid.methods}")
    print(f"  Vocab sizes : {grid.vocab_sizes}")
    print(f"  Algorithms  : {grid.algorithms}")
    print("=" * 80)

    t0 = time.time()
    runner = ExperimentRunner(grid)
    results = runner.run()
    elapsed = time.time() - t0

    print(f"\n✓ {len(results)} deney tamamlandı  |  süre: {elapsed/60:.1f} dakika")

    reporter = Reporter()
    reporter.save(results)

    print(f"\n✓ Çalıştırma #{reporter.run_no} bitti.")
    print(f"  → {reporter.run_dir}")


if __name__ == "__main__":
    main()
