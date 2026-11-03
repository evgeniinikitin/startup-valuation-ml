"""Run the trained valuation model on one feature record."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from modular_rag.ml_valuation import predict_ml_from_feature_row  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Estimate pre-money valuation in USD")
    parser.add_argument("input", type=Path, help="JSON object with startup features")
    args = parser.parse_args()
    features = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(features, dict):
        parser.error("input must be a JSON object")
    result = predict_ml_from_feature_row(features)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
