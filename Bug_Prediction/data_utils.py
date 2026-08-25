"""Shared data validation and duplicate-handling helpers for bug prediction."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


FEATURES = [
    "cbo", "dit", "fanIn", "fanOut", "lcom", "noc", "numberOfAttributes",
    "numberOfLinesOfCode", "numberOfMethods", "numberOfPrivateAttributes",
    "numberOfPrivateMethods", "numberOfPublicAttributes", "numberOfPublicMethods",
    "rfc", "wmc", "numberOfVersionsUntil", "numberOfAuthorsUntil", "linesAddedUntil",
    "maxLinesAddedUntil", "avgLinesAddedUntil", "linesRemovedUntil",
    "maxLinesRemovedUntil", "avgLinesRemovedUntil", "codeChurnUntil",
    "maxCodeChurnUntil", "avgCodeChurnUntil", "ageWithRespectTo",
    "weightedAgeWithRespectTo",
]
TARGET = "defect"
REQUIRED_COLUMNS = ["project", "classname", *FEATURES, "bugs", TARGET]
FEATURE_ALIASES = {"fanin": "fanIn", "fanout": "fanOut"}


def validate_dataset(frame: pd.DataFrame) -> None:
    """Ensure the supplied AEEEM CSV has every column needed by this baseline."""
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    if not set(frame[TARGET].dropna().unique()).issubset({0, 1}):
        raise ValueError("The defect target must contain only binary 0/1 labels.")


def class_distribution(labels: pd.Series) -> dict[str, Any]:
    total = int(len(labels))
    defective = int((labels == 1).sum())
    non_defective = int((labels == 0).sum())
    return {
        "total": total,
        "defective": defective,
        "non_defective": non_defective,
        "defective_percentage": round((defective / total * 100) if total else 0.0, 4),
        "non_defective_percentage": round((non_defective / total * 100) if total else 0.0, 4),
    }


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    return value


def audit_dataset(frame: pd.DataFrame) -> dict[str, Any]:
    """Return the complete pre-training audit required for the AEEEM data."""
    validate_dataset(frame)
    numeric_features = frame[FEATURES].apply(pd.to_numeric, errors="raise")
    missing_by_column = {
        column: int(count) for column, count in frame.isna().sum().items() if count
    }
    duplicate_count = int(frame.duplicated(subset=FEATURES, keep="first").sum())
    return {
        "original_row_count": int(len(frame)),
        "input_feature_count": len(FEATURES),
        "project_count": int(frame["project"].nunique(dropna=True)),
        "missing_value_count": int(frame.isna().sum().sum()),
        "missing_values_by_column": missing_by_column,
        "infinite_value_count": int(np.isinf(numeric_features.to_numpy()).sum()),
        "duplicate_feature_rows": duplicate_count,
        "original_class_distribution": class_distribution(frame[TARGET]),
    }


def find_duplicate_conflicts(frame: pd.DataFrame) -> tuple[pd.Index, list[dict[str, Any]]]:
    """Find feature vectors that have more than one defect label.

    The returned index contains every source row belonging to a conflicting vector.
    Keeping this inspection separate prevents a label from being silently discarded.
    """
    grouped = frame.groupby(FEATURES, dropna=False, sort=False)
    conflict_indexes: list[Any] = []
    conflicts: list[dict[str, Any]] = []
    for feature_values, group in grouped:
        labels = sorted(int(label) for label in group[TARGET].unique())
        if len(labels) > 1:
            source_indexes = [_to_builtin(index) for index in group.index.tolist()]
            conflict_indexes.extend(group.index.tolist())
            conflicts.append(
                {
                    "feature_values": {
                        feature: _to_builtin(value)
                        for feature, value in zip(FEATURES, feature_values, strict=True)
                    },
                    "row_count": int(len(group)),
                    "defect_labels": labels,
                    "source_row_indexes": source_indexes,
                }
            )
    return pd.Index(conflict_indexes), conflicts


def clean_dataset(
    frame: pd.DataFrame,
    *,
    conflict_policy: str = "exclude",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Remove duplicate feature observations with an explicit conflict policy.

    ``exclude`` is the default because contradictory labels provide no single target for
    an identical feature vector and retaining them could put the same vector in both
    split partitions. Every excluded conflicting vector is included in the audit report.
    """
    if conflict_policy not in {"exclude", "retain"}:
        raise ValueError("conflict_policy must be either 'exclude' or 'retain'.")

    conflict_indexes, conflicts = find_duplicate_conflicts(frame)
    conflict_index_set = set(conflict_indexes.tolist())
    non_conflicting = frame.loc[~frame.index.isin(conflict_index_set)]
    deduplicated = non_conflicting.drop_duplicates(subset=FEATURES, keep="first")

    if conflict_policy == "exclude":
        cleaned = deduplicated
        excluded_conflict_rows = len(conflict_indexes)
    else:
        # This option exists for data investigation only. It intentionally retains all
        # contradictory rows and is recorded prominently in the metadata.
        cleaned = pd.concat([deduplicated, frame.loc[conflict_indexes]]).sort_index()
        excluded_conflict_rows = 0

    cleaned = cleaned.reset_index(drop=True)
    details = {
        "conflict_policy": conflict_policy,
        "conflicting_feature_vectors": len(conflicts),
        "conflicting_affected_rows": int(len(conflict_indexes)),
        "conflicting_duplicate_details": conflicts,
        "valid_duplicate_rows_removed": int(len(non_conflicting) - len(deduplicated)),
        "conflicting_rows_excluded": int(excluded_conflict_rows),
        "cleaned_row_count": int(len(cleaned)),
        "cleaned_class_distribution": class_distribution(cleaned[TARGET]),
    }
    return cleaned, details


def normalise_feature_mapping(values: dict[str, Any], schema: Iterable[str]) -> pd.DataFrame:
    """Map analysis-worker ``fanin``/``fanout`` inputs to the dataset naming."""
    normalised = {FEATURE_ALIASES.get(key, key): value for key, value in values.items()}
    expected = list(schema)
    missing = [feature for feature in expected if feature not in normalised]
    extras = [key for key in normalised if key not in expected]
    if missing or extras:
        raise ValueError(f"Expected exactly the model schema; missing={missing}, extras={extras}")
    return pd.DataFrame([[normalised[feature] for feature in expected]], columns=expected)
