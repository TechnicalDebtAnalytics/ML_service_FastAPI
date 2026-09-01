"""Preprocessing and normalization for Bug Prediction class metrics."""

from typing import Any
import pandas as pd


# Canonical 28 features expected by the trained XGBoost model
CANONICAL_FEATURES = [
    "cbo", "dit", "fanIn", "fanOut", "lcom", "noc", "numberOfAttributes",
    "numberOfLinesOfCode", "numberOfMethods", "numberOfPrivateAttributes",
    "numberOfPrivateMethods", "numberOfPublicAttributes", "numberOfPublicMethods",
    "rfc", "wmc", "numberOfVersionsUntil", "numberOfAuthorsUntil", "linesAddedUntil",
    "maxLinesAddedUntil", "avgLinesAddedUntil", "linesRemovedUntil",
    "maxLinesRemovedUntil", "avgLinesRemovedUntil", "codeChurnUntil",
    "maxCodeChurnUntil", "avgCodeChurnUntil", "ageWithRespectTo",
    "weightedAgeWithRespectTo",
]

# Alias map to support both snake_case (DB / REST) and camelCase (Java DTOs)
FEATURE_ALIASES: dict[str, str] = {
    # Fan in/out
    "fan_in": "fanIn",
    "fanin": "fanIn",
    "fan_out": "fanOut",
    "fanout": "fanOut",

    # Size and Visibility
    "number_of_attributes": "numberOfAttributes",
    "numberofattributes": "numberOfAttributes",
    "number_of_lines_of_code": "numberOfLinesOfCode",
    "numberoflinesofcode": "numberOfLinesOfCode",
    "number_of_methods": "numberOfMethods",
    "numberofmethods": "numberOfMethods",
    "number_of_private_attributes": "numberOfPrivateAttributes",
    "numberofprivateattributes": "numberOfPrivateAttributes",
    "number_of_private_methods": "numberOfPrivateMethods",
    "numberofprivatemethods": "numberOfPrivateMethods",
    "number_of_public_attributes": "numberOfPublicAttributes",
    "numberofpublicattributes": "numberOfPublicAttributes",
    "number_of_public_methods": "numberOfPublicMethods",
    "numberofpublicmethods": "numberOfPublicMethods",

    # Git Churn & Author Metrics
    "number_of_versions_until": "numberOfVersionsUntil",
    "numberofversionsuntil": "numberOfVersionsUntil",
    "number_of_authors_until": "numberOfAuthorsUntil",
    "numberofauthorsuntil": "numberOfAuthorsUntil",
    "lines_added_until": "linesAddedUntil",
    "linesaddeduntil": "linesAddedUntil",
    "max_lines_added_until": "maxLinesAddedUntil",
    "maxlinesaddeduntil": "maxLinesAddedUntil",
    "avg_lines_added_until": "avgLinesAddedUntil",
    "avglinesaddeduntil": "avgLinesAddedUntil",
    "lines_removed_until": "linesRemovedUntil",
    "linesremoveduntil": "linesRemovedUntil",
    "max_lines_removed_until": "maxLinesRemovedUntil",
    "maxlinesremoveduntil": "maxLinesRemovedUntil",
    "avg_lines_removed_until": "avgLinesRemovedUntil",
    "avglinesremoveduntil": "avgLinesRemovedUntil",
    "code_churn_until": "codeChurnUntil",
    "codechurnuntil": "codeChurnUntil",
    "max_code_churn_until": "maxCodeChurnUntil",
    "maxcodechurnuntil": "maxCodeChurnUntil",
    "avg_code_churn_until": "avgCodeChurnUntil",
    "avgcodechurnuntil": "avgCodeChurnUntil",
    "age_with_respect_to": "ageWithRespectTo",
    "agewithrespectto": "ageWithRespectTo",
    "weighted_age_with_respect_to": "weightedAgeWithRespectTo",
    "weightedagewithrespectto": "weightedAgeWithRespectTo",
}


def normalize_class_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    """Normalize input metric keys to canonical feature names with default fallbacks."""
    normalized: dict[str, float] = {}

    for key, value in metrics.items():
        canonical_key = FEATURE_ALIASES.get(key, FEATURE_ALIASES.get(key.lower(), key))
        if canonical_key in CANONICAL_FEATURES:
            try:
                normalized[canonical_key] = float(value) if value is not None else 0.0
            except (ValueError, TypeError):
                normalized[canonical_key] = 0.0

    # Ensure all 28 canonical features are present with 0.0 fallback if missing
    for feature in CANONICAL_FEATURES:
        if feature not in normalized:
            normalized[feature] = 0.0

    return normalized


def build_bug_feature_dataframe(metrics: dict[str, Any]) -> pd.DataFrame:
    """Construct a single-row DataFrame ordered by CANONICAL_FEATURES for XGBoost prediction."""
    normalized = normalize_class_metrics(metrics)
    row = [normalized[feature] for feature in CANONICAL_FEATURES]
    return pd.DataFrame([row], columns=CANONICAL_FEATURES)
