"""Evaluate the bundled model against a private CSV with known USD valuations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from modular_rag.ml_batch_eval import evaluate_ml_csv  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate valuation predictions")
    parser.add_argument("csv", type=Path, help="CSV with feature columns and pre-money USD target")
    parser.add_argument("--target-col", default=None)
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()
    metrics, _ = evaluate_ml_csv(
        pd.read_csv(args.csv, low_memory=False),
        target_col=args.target_col,
        max_rows=args.max_rows,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
