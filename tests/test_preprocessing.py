"""Tests for SATD and Bug prediction preprocessors."""

from unittest.mock import Mock, patch

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from app.preprocessing.satd_preprocessor import (
    build_satd_feature_vector,
    clean_comment,
    extract_satd_keyword_features,
)
from app.preprocessing.bug_preprocessor import (
    CANONICAL_FEATURES,
    build_bug_feature_dataframe,
    normalize_class_metrics,
)


def test_clean_comment():
    dirty = "/* TODO: fix this memory leak http://example.com/bug/123 */"
    cleaned = clean_comment(dirty)
    assert "http" not in cleaned
    assert "/*" not in cleaned
    assert "todo" in cleaned
    assert "memory leak" in cleaned


def test_normalize_class_metrics_with_aliases():
    input_metrics = {
        "fan_in": 5,
        "fan_out": 3,
        "number_of_attributes": 10,
        "numberOfLinesOfCode": 100,
        "wmc": 4.5,
        "cbo": 2,
    }
    normalized = normalize_class_metrics(input_metrics)

    # Verify all 28 canonical features are present
    for feature in CANONICAL_FEATURES:
        assert feature in normalized

    assert normalized["fanIn"] == 5.0
    assert normalized["fanOut"] == 3.0
    assert normalized["numberOfAttributes"] == 10.0
    assert normalized["numberOfLinesOfCode"] == 100.0
    assert normalized["wmc"] == 4.5
    assert normalized["cbo"] == 2.0
    assert normalized["dit"] == 0.0  # default fallback


@pytest.mark.parametrize("value", [None, "", 0, False])
def test_clean_comment_returns_empty_string_for_falsy_input(value):
    assert clean_comment(value) == ""


@pytest.mark.parametrize(
    ("dirty", "expected"),
    [
        ("// MIXED_case  Value-42!!!\n\t*", "mixed_case value 42"),
        ("Read HTTP://EXAMPLE.COM/path now", "read now"),
        ("/***/", ""),
        (12345, "12345"),
    ],
)
def test_clean_comment_normalizes_case_symbols_whitespace_and_non_string_values(dirty, expected):
    assert clean_comment(dirty) == expected


def test_extract_satd_keyword_features_preserves_category_order_and_weights():
    feature_dictionary = {
        "DESIGN": {"todo": 2.0, "fix later": 1.0},
        "TEST": {"missing test": 0.5},
    }

    result = extract_satd_keyword_features("todo fix later and add missing test", feature_dictionary)

    # Weighted category scores come first, followed by one binary flag per category.
    assert result == [15.0, 2.5, 1.0, 1.0]


def test_extract_satd_keyword_features_returns_zeros_for_no_matches_and_empty_for_no_categories():
    feature_dictionary = {
        "DESIGN": {"todo": 2.0},
        "TEST": {"missing test": 0.5},
    }

    assert extract_satd_keyword_features("ordinary explanatory comment", feature_dictionary) == [0.0, 0.0, 0.0, 0.0]
    assert extract_satd_keyword_features("anything", {}) == []


def test_extract_satd_keyword_features_propagates_malformed_weight_error():
    with pytest.raises(TypeError):
        extract_satd_keyword_features("todo", {"DESIGN": {"todo": "heavy"}})


def test_build_satd_feature_vector_combines_word_char_and_keyword_features_in_order():
    word_vectorizer = Mock()
    char_vectorizer = Mock()
    word_vectorizer.transform.return_value = csr_matrix([[1.0, 2.0]])
    char_vectorizer.transform.return_value = csr_matrix([[3.0]])
    feature_dictionary = {
        "DESIGN": {"todo": 2.0},
        "TEST": {"missing test": 1.0},
    }

    result = build_satd_feature_vector(
        "todo and missing test", word_vectorizer, char_vectorizer, feature_dictionary
    )

    word_vectorizer.transform.assert_called_once_with(["todo and missing test"])
    char_vectorizer.transform.assert_called_once_with(["todo and missing test"])
    assert result.shape == (1, 7)
    np.testing.assert_array_equal(
        result.toarray(),
        [[1.0, 2.0, 3.0, 10.0, 5.0, 1.0, 1.0]],
    )


def test_build_satd_feature_vector_supports_empty_comment_and_feature_dictionary():
    word_vectorizer = Mock()
    char_vectorizer = Mock()
    word_vectorizer.transform.return_value = csr_matrix([[0.0]])
    char_vectorizer.transform.return_value = csr_matrix([[0.0, 0.0]])

    result = build_satd_feature_vector("", word_vectorizer, char_vectorizer, {})

    assert result.shape == (1, 3)
    np.testing.assert_array_equal(result.toarray(), [[0.0, 0.0, 0.0]])


def test_build_satd_feature_vector_propagates_vectorizer_failure_and_stops():
    word_vectorizer = Mock()
    char_vectorizer = Mock()
    word_vectorizer.transform.side_effect = ValueError("word vectorizer failed")

    with pytest.raises(ValueError, match="word vectorizer failed"):
        build_satd_feature_vector("todo", word_vectorizer, char_vectorizer, {})

    char_vectorizer.transform.assert_not_called()


def test_normalize_class_metrics_empty_input_returns_all_features_in_canonical_order():
    normalized = normalize_class_metrics({})

    assert list(normalized) == CANONICAL_FEATURES
    assert len(normalized) == 28
    assert all(value == 0.0 for value in normalized.values())
    assert len(set(CANONICAL_FEATURES)) == len(CANONICAL_FEATURES)


def test_normalize_class_metrics_coerces_values_and_defaults_invalid_values():
    normalized = normalize_class_metrics(
        {
            "cbo": "12",
            "dit": None,
            "lcom": "not-a-number",
            "noc": object(),
            "rfc": True,
            "wmc": 4.25,
        }
    )

    assert normalized["cbo"] == 12.0
    assert normalized["dit"] == 0.0
    assert normalized["lcom"] == 0.0
    assert normalized["noc"] == 0.0
    assert normalized["rfc"] == 1.0
    assert normalized["wmc"] == 4.25


def test_normalize_class_metrics_handles_case_insensitive_aliases_and_ignores_unknown_fields():
    normalized = normalize_class_metrics(
        {
            "FAN_IN": 7,
            "NUMBEROFLINESOFCODE": 100,
            "unrelatedMetadata": 999,
        }
    )

    assert normalized["fanIn"] == 7.0
    assert normalized["numberOfLinesOfCode"] == 100.0
    assert "unrelatedMetadata" not in normalized


def test_normalize_class_metrics_uses_last_value_when_aliases_target_same_feature():
    normalized = normalize_class_metrics({"fan_in": 2, "fanin": 3, "fanIn": 4})

    assert normalized["fanIn"] == 4.0


def test_normalize_class_metrics_preserves_numeric_boundary_values_without_clamping():
    normalized = normalize_class_metrics(
        {
            "cbo": -1,
            "dit": 0,
            "wmc": 1.0e100,
            "lcom": float("inf"),
        }
    )

    assert normalized["cbo"] == -1.0
    assert normalized["dit"] == 0.0
    assert normalized["wmc"] == 1.0e100
    assert normalized["lcom"] == float("inf")


@pytest.mark.parametrize("malformed", [None, [], "metrics"])
def test_normalize_class_metrics_rejects_non_mapping_input(malformed):
    with pytest.raises(AttributeError):
        normalize_class_metrics(malformed)


def test_normalize_class_metrics_rejects_non_string_metric_keys():
    with pytest.raises(AttributeError):
        normalize_class_metrics({1: 10})


def test_build_bug_feature_dataframe_has_one_row_and_exact_canonical_column_order():
    metrics = {feature: index + 0.5 for index, feature in enumerate(reversed(CANONICAL_FEATURES))}

    dataframe = build_bug_feature_dataframe(metrics)

    assert dataframe.shape == (1, len(CANONICAL_FEATURES))
    assert dataframe.columns.tolist() == CANONICAL_FEATURES
    assert dataframe.iloc[0].tolist() == [metrics[feature] for feature in CANONICAL_FEATURES]


def test_build_bug_feature_dataframe_defaults_all_missing_metrics_to_zero():
    dataframe = build_bug_feature_dataframe({})

    assert dataframe.columns.tolist() == CANONICAL_FEATURES
    assert dataframe.iloc[0].tolist() == [0.0] * len(CANONICAL_FEATURES)


def test_build_bug_feature_dataframe_propagates_normalization_failure():
    with (
        patch(
            "app.preprocessing.bug_preprocessor.normalize_class_metrics",
            side_effect=TypeError("normalization failed"),
        ),
        pytest.raises(TypeError, match="normalization failed"),
    ):
        build_bug_feature_dataframe({"cbo": 1})
