"""Preprocessing logic for Self-Admitted Technical Debt (SATD) comments."""

import re
from typing import Any
from scipy.sparse import csr_matrix, hstack


def clean_comment(text: str | None) -> str:
    """Clean and normalize source code comment text."""
    if not text:
        return ""

    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+", " ", text)

    # Remove comment syntax markers: //, /*, */, *
    text = re.sub(r"//|/\*|\*/|\*", " ", text)

    # Remove non-alphanumeric characters except underscore and whitespace
    text = re.sub(r"[^a-z0-9_\s]", " ", text)

    # Normalize multiple whitespace characters
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def extract_satd_keyword_features(text: str, satd_features_dict: dict[str, dict[str, float]]) -> list[float]:
    """Extract weighted category scores and binary keyword indicators from cleaned comment."""
    features: list[float] = []

    # 1. Weighted scores per category
    for category, keywords in satd_features_dict.items():
        score = 0.0
        for phrase, weight in keywords.items():
            if phrase in text:
                score += weight * 5.0
        features.append(score)

    # 2. Binary indicators per category
    for category, keywords in satd_features_dict.items():
        found = 0.0
        for phrase in keywords:
            if phrase in text:
                found = 1.0
                break
        features.append(found)

    return features


def build_satd_feature_vector(
    cleaned_comment: str,
    word_vectorizer: Any,
    char_vectorizer: Any,
    satd_features_dict: dict[str, dict[str, float]],
) -> Any:
    """Build the combined sparse feature matrix for the SATD LinearSVC model."""
    word_features = word_vectorizer.transform([cleaned_comment])
    char_features = char_vectorizer.transform([cleaned_comment])
    satd_keyword_features = csr_matrix([extract_satd_keyword_features(cleaned_comment, satd_features_dict)])

    final_features = hstack([word_features, char_features, satd_keyword_features])
    return final_features
