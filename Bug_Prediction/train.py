"""Train the reproducible baseline XGBoost bug-prediction model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from data_utils import FEATURES, TARGET, audit_dataset, class_distribution, clean_dataset


MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = MODULE_DIR / "dataset" / "AEEEM_reduced_28_features.csv"
MODEL_DIR = MODULE_DIR / "model"
MODEL_PATH = MODEL_DIR / "xgboost_bug_prediction.json"
SCHEMA_PATH = MODEL_DIR / "feature_schema.json"
METADATA_PATH = MODEL_DIR / "model_metadata.json"
REPORT_PATH = MODEL_DIR / "training_report.json"

# Deliberately fixed baseline configuration: no tuning, balancing, or threshold changes.
XGBOOST_PARAMETERS: dict[str, Any] = {
    "objective": "binary:logistic",
    "n_estimators": 100,
    "learning_rate": 0.1,
    "max_depth": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "eval_metric": "logloss",
    "n_jobs": 1,
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _evaluate(model: XGBClassifier, features: pd.DataFrame, labels: pd.Series) -> tuple[dict[str, float], list[list[int]]]:
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(labels, predicted)),
        "precision": float(precision_score(labels, predicted, zero_division=0)),
        "recall": float(recall_score(labels, predicted, zero_division=0)),
        "f1_score": float(f1_score(labels, predicted, zero_division=0)),
        "roc_auc": float(roc_auc_score(labels, probabilities)),
        "pr_auc": float(average_precision_score(labels, probabilities)),
        "mcc": float(matthews_corrcoef(labels, predicted)),
    }
    return metrics, confusion_matrix(labels, predicted, labels=[0, 1]).tolist()


def _print_distribution(title: str, distribution: dict[str, Any]) -> None:
    print(
        f"{title}: total={distribution['total']}, defective={distribution['defective']} "
        f"({distribution['defective_percentage']:.2f}%), non-defective="
        f"{distribution['non_defective']} ({distribution['non_defective_percentage']:.2f}%)"
    )


def train(
    dataset_path: Path = DEFAULT_DATASET,
    *,
    random_state: int = 42,
    test_size: float = 0.20,
    conflict_policy: str = "exclude",
) -> dict[str, Any]:
    """Run the entire audit, clean, split, train, evaluation, and save workflow."""
    frame = pd.read_csv(dataset_path)
    audit = audit_dataset(frame)
    if audit["infinite_value_count"]:
        raise ValueError("The dataset contains infinite input-feature values.")
    if audit["missing_value_count"]:
        raise ValueError("The dataset contains missing values; resolve them before training.")

    cleaned, duplicate_report = clean_dataset(frame, conflict_policy=conflict_policy)
    train_frame, test_frame = train_test_split(
        cleaned,
        test_size=test_size,
        stratify=cleaned[TARGET],
        random_state=random_state,
    )
    x_train, y_train = train_frame[FEATURES], train_frame[TARGET].astype(int)
    x_test, y_test = test_frame[FEATURES], test_frame[TARGET].astype(int)

    model = XGBClassifier(random_state=random_state, **XGBOOST_PARAMETERS)
    model.fit(x_train, y_train)
    metrics, matrix = _evaluate(model, x_test, y_test)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save_model(MODEL_PATH)
    schema = {
        "feature_names": FEATURES,
        "feature_count": len(FEATURES),
        "feature_aliases": {"fanin": "fanIn", "fanout": "fanOut"},
    }
    _write_json(SCHEMA_PATH, schema)

    train_distribution = class_distribution(y_train)
    test_distribution = class_distribution(y_test)
    metadata = {
        "model_name": "XGBoost Bug Prediction Classifier",
        "dataset_name": dataset_path.name,
        "original_row_count": audit["original_row_count"],
        "duplicate_count": audit["duplicate_feature_rows"],
        "valid_duplicate_rows_removed": duplicate_report["valid_duplicate_rows_removed"],
        "conflict_policy": conflict_policy,
        "conflicting_feature_vectors": duplicate_report["conflicting_feature_vectors"],
        "conflicting_affected_rows": duplicate_report["conflicting_affected_rows"],
        "conflicting_rows_excluded": duplicate_report["conflicting_rows_excluded"],
        "cleaned_row_count": duplicate_report["cleaned_row_count"],
        "feature_count": len(FEATURES),
        "feature_names": FEATURES,
        "target_name": TARGET,
        "train_size": int(len(train_frame)),
        "test_size": int(len(test_frame)),
        "test_size_fraction": test_size,
        "train_defective_count": train_distribution["defective"],
        "train_non_defective_count": train_distribution["non_defective"],
        "test_defective_count": test_distribution["defective"],
        "test_non_defective_count": test_distribution["non_defective"],
        "random_state": random_state,
        "xgboost_parameters": {**XGBOOST_PARAMETERS, "random_state": random_state},
        "warnings": (
            [
                "Contradictory labels were found for identical feature vectors; "
                "the complete vectors were excluded before splitting."
            ]
            if duplicate_report["conflicting_feature_vectors"]
            else []
        ),
    }
    _write_json(METADATA_PATH, metadata)

    report = {
        "dataset_audit": audit,
        "duplicate_handling": duplicate_report,
        "split_summary": {
            "cleaned": duplicate_report["cleaned_class_distribution"],
            "training": train_distribution,
            "testing": test_distribution,
        },
        "evaluation": {"metrics": metrics, "confusion_matrix": matrix},
        "warnings": metadata["warnings"],
        "artifacts": {
            "model_path": str(MODEL_PATH),
            "feature_schema_path": str(SCHEMA_PATH),
            "metadata_path": str(METADATA_PATH),
        },
    }
    _write_json(REPORT_PATH, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument(
        "--conflict-policy",
        choices=("exclude", "retain"),
        default="exclude",
        help="How to handle feature vectors with contradictory defect labels (default: exclude).",
    )
    args = parser.parse_args()
    report = train(
        args.dataset,
        random_state=args.random_state,
        test_size=args.test_size,
        conflict_policy=args.conflict_policy,
    )

    audit = report["dataset_audit"]
    duplicates = report["duplicate_handling"]
    print("Dataset audit")
    print(
        f"Original rows={audit['original_row_count']}; features={audit['input_feature_count']}; "
        f"projects={audit['project_count']}; missing={audit['missing_value_count']}; "
        f"infinite={audit['infinite_value_count']}; duplicate feature rows={audit['duplicate_feature_rows']}"
    )
    _print_distribution("Original", audit["original_class_distribution"])
    print(
        f"Conflicting feature vectors={duplicates['conflicting_feature_vectors']}; "
        f"affected rows={duplicates['conflicting_affected_rows']}; policy={duplicates['conflict_policy']}"
    )
    for number, conflict in enumerate(duplicates["conflicting_duplicate_details"], start=1):
        print(
            f"Conflict {number}: rows={conflict['source_row_indexes']}; "
            f"defect labels={conflict['defect_labels']}; "
            f"feature vector={conflict['feature_values']}"
        )
    _print_distribution("Cleaned", report["split_summary"]["cleaned"])
    _print_distribution("Training", report["split_summary"]["training"])
    _print_distribution("Testing", report["split_summary"]["testing"])
    print("Evaluation:", json.dumps(report["evaluation"], indent=2))
    print("Saved model:", report["artifacts"]["model_path"])
    print("Saved feature schema:", report["artifacts"]["feature_schema_path"])
    print("Saved metadata:", report["artifacts"]["metadata_path"])


if __name__ == "__main__":
    main()
