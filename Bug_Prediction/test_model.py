"""Load the saved XGBoost model and predict one cleaned-dataset sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from xgboost import XGBClassifier

from data_utils import clean_dataset, normalise_feature_mapping


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = MODULE_DIR / "dataset" / "AEEEM_reduced_28_features.csv"
MODEL_DIR = MODULE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgboost_bug_prediction.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"


def _load_schema() -> list[str]:
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    feature_names = schema["feature_names"] if isinstance(schema, dict) else schema
    if len(feature_names) != 28 or len(set(feature_names)) != 28:
        raise ValueError("feature_schema.json must define exactly 28 unique model features.")
    return feature_names


def _load_model() -> XGBClassifier:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Saved model not found: {MODEL_PATH}. Run train.py first.")
    model = XGBClassifier()
    model.load_model(MODEL_PATH)
    return model


def predict_feature_mapping(values: dict[str, Any]) -> tuple[int, float]:
    """Predict an analysis-worker feature mapping, accepting fanin/fanout aliases."""
    schema = _load_schema()
    row = normalise_feature_mapping(values, schema)
    model = _load_model()
    prediction = int(model.predict(row)[0])
    probability = float(model.predict_proba(row)[0, 1])
    return prediction, probability


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--row", type=int, default=0, help="Cleaned dataset row to predict.")
    parser.add_argument(
        "--input-json",
        type=Path,
        help="Optional JSON object containing exactly the 28 features; fanin/fanout are accepted.",
    )
    args = parser.parse_args()

    if args.input_json:
        values = json.loads(args.input_json.read_text(encoding="utf-8"))
    else:
        metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
        cleaned, _ = clean_dataset(
            pd.read_csv(args.dataset), conflict_policy=metadata.get("conflict_policy", "exclude")
        )
        if args.row < 0 or args.row >= len(cleaned):
            raise IndexError(f"--row must be between 0 and {len(cleaned) - 1}.")
        schema = _load_schema()
        values = cleaned.loc[args.row, schema].to_dict()

    predicted_class, defect_probability = predict_feature_mapping(values)
    print(f"predicted class: {predicted_class}")
    print(f"defect probability: {defect_probability:.6f}")


if __name__ == "__main__":
    main()
