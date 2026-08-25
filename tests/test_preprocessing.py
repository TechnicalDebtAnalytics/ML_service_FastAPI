"""Tests for SATD and Bug prediction preprocessors."""

from app.preprocessing.satd_preprocessor import clean_comment
from app.preprocessing.bug_preprocessor import normalize_class_metrics, CANONICAL_FEATURES


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
