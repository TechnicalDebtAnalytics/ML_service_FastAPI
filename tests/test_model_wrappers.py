"""Unit tests for SATD (TF-IDF + XGBoost) and Bug Prediction model wrappers."""

import json
from unittest.mock import Mock, patch
import numpy as np
import pytest

from app.config.settings import settings
from app.models.bug_model_wrapper import BugModelWrapper
from app.models.satd_model_wrapper import SATDModelWrapper
from app.preprocessing.bug_preprocessor import CANONICAL_FEATURES


# ---------------------------------------------------------------------------
# SATDModelWrapper Tests (TF-IDF + XGBoost)
# ---------------------------------------------------------------------------

def test_satd_load_reads_vectorizer_model_and_classes(tmp_path) -> None:
    vec_path = tmp_path / settings.SATD_VECTORIZER_FILE
    model_path = tmp_path / settings.SATD_MODEL_FILE
    classes_path = tmp_path / settings.SATD_CLASSES_FILE

    vec_path.touch()
    model_path.touch()
    classes_payload = {"id2label": {"0": "code/design_debt", "1": "non_debt"}}
    classes_path.write_text(json.dumps(classes_payload), encoding="utf-8")

    classifier = Mock()
    vectorizer_mock = Mock()
    wrapper = SATDModelWrapper(tmp_path)

    with (
        patch("app.models.satd_model_wrapper.joblib.load", return_value=vectorizer_mock) as mock_joblib,
        patch("app.models.satd_model_wrapper.xgb.XGBClassifier", return_value=classifier) as classifier_type,
    ):
        wrapper.load()

    mock_joblib.assert_called_once_with(str(vec_path))
    classifier_type.assert_called_once_with()
    classifier.load_model.assert_called_once_with(str(model_path))
    assert wrapper.is_loaded is True
    assert wrapper.id2label == {0: "code/design_debt", 1: "non_debt"}


def test_satd_load_rejects_missing_vectorizer_or_model(tmp_path) -> None:
    wrapper = SATDModelWrapper(tmp_path)
    with pytest.raises(FileNotFoundError, match="SATD TF-IDF vectorizer not found"):
        wrapper.load()

    # Now create vectorizer but omit model
    (tmp_path / settings.SATD_VECTORIZER_FILE).touch()
    with patch("app.models.satd_model_wrapper.joblib.load", return_value=Mock()):
        with pytest.raises(FileNotFoundError, match="SATD XGBoost model artifact not found"):
            wrapper.load()


def test_satd_predict_single_comment() -> None:
    wrapper = SATDModelWrapper()
    wrapper.is_loaded = True
    wrapper.vectorizer = Mock()
    wrapper.model = Mock()
    wrapper.id2label = {0: "code/design_debt", 1: "non_debt"}

    wrapper.vectorizer.transform.return_value = object()
    wrapper.model.predict_proba.return_value = np.array([[0.85, 0.15]])

    res = wrapper.predict_comment("TODO: refactor this method")
    assert res["comment"] == "TODO: refactor this method"
    assert res["category"] == "code/design_debt"
    assert res["confidence_score"] == 0.85
    assert res["is_debt"] is True


@pytest.mark.parametrize("label,expected_debt", [
    ("non_debt", False),
    ("clean", False),
    ("without_classification", False),
    ("code/design_debt", True),
    ("defect_debt", True),
])
def test_satd_predict_maps_debt_status(label: str, expected_debt: bool) -> None:
    wrapper = SATDModelWrapper()
    wrapper.is_loaded = True
    wrapper.vectorizer = Mock()
    wrapper.model = Mock()
    wrapper.id2label = {0: label}

    wrapper.vectorizer.transform.return_value = object()
    wrapper.model.predict_proba.return_value = np.array([[0.90]])

    res = wrapper.predict_comment("sample text")
    assert res["category"] == label
    assert res["is_debt"] is expected_debt


def test_satd_predict_empty_or_none_comment() -> None:
    wrapper = SATDModelWrapper()
    wrapper.is_loaded = True
    wrapper.vectorizer = Mock()
    wrapper.model = Mock()
    wrapper.id2label = {0: "non_debt"}

    res = wrapper.predict_comment("")
    assert res["category"] == "non_debt"
    assert res["confidence_score"] == 1.0
    assert res["is_debt"] is False


def test_satd_predict_rejects_unloaded() -> None:
    wrapper = SATDModelWrapper()
    wrapper.is_loaded = False
    with pytest.raises(RuntimeError, match="SATD model is not loaded"):
        wrapper.predict_comments(["hello"])


# ---------------------------------------------------------------------------
# BugModelWrapper Tests (XGBoost)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    ("schema_payload", "expected_schema"),
    [
        (["cbo", "wmc"], ["cbo", "wmc"]),
        ({"features": ["dit", "rfc"]}, ["dit", "rfc"]),
    ],
)
def test_bug_load_reads_model_and_supported_schema_formats(tmp_path, schema_payload, expected_schema) -> None:
    model_path = tmp_path / settings.BUG_MODEL_FILE
    model_path.touch()
    (tmp_path / settings.BUG_SCHEMA_FILE).write_text(json.dumps(schema_payload), encoding="utf-8")
    classifier = Mock()
    wrapper = BugModelWrapper(tmp_path)

    with patch("app.models.bug_model_wrapper.xgb.XGBClassifier", return_value=classifier) as classifier_type:
        wrapper.load()

    classifier_type.assert_called_once_with()
    classifier.load_model.assert_called_once_with(str(model_path))
    assert wrapper.model is classifier
    assert wrapper.feature_schema == expected_schema
    assert wrapper.is_loaded is True


def test_bug_load_uses_canonical_schema_when_schema_is_missing_or_corrupt(tmp_path) -> None:
    model_path = tmp_path / settings.BUG_MODEL_FILE
    model_path.touch()
    classifier = Mock()
    wrapper_without_schema = BugModelWrapper(tmp_path)

    with patch("app.models.bug_model_wrapper.xgb.XGBClassifier", return_value=classifier):
        wrapper_without_schema.load()
    assert wrapper_without_schema.feature_schema == CANONICAL_FEATURES

    (tmp_path / settings.BUG_SCHEMA_FILE).write_text("{not-json", encoding="utf-8")
    wrapper_with_corrupt_schema = BugModelWrapper(tmp_path)
    with patch("app.models.bug_model_wrapper.xgb.XGBClassifier", return_value=Mock()):
        wrapper_with_corrupt_schema.load()
    assert wrapper_with_corrupt_schema.feature_schema == CANONICAL_FEATURES


def test_bug_load_rejects_missing_model_file(tmp_path) -> None:
    wrapper = BugModelWrapper(tmp_path)

    with (
        patch("app.models.bug_model_wrapper.xgb.XGBClassifier") as classifier_type,
        pytest.raises(FileNotFoundError, match="Bug prediction model artifact not found"),
    ):
        wrapper.load()

    classifier_type.assert_not_called()
    assert wrapper.is_loaded is False


def test_bug_load_propagates_corrupt_model_failure_and_stays_unloaded(tmp_path) -> None:
    (tmp_path / settings.BUG_MODEL_FILE).touch()
    classifier = Mock()
    classifier.load_model.side_effect = ValueError("invalid XGBoost model")
    wrapper = BugModelWrapper(tmp_path)

    with (
        patch("app.models.bug_model_wrapper.xgb.XGBClassifier", return_value=classifier),
        pytest.raises(ValueError, match="invalid XGBoost model"),
    ):
        wrapper.load()

    assert wrapper.model is None
    assert wrapper.is_loaded is False


def test_bug_predict_rejects_unloaded_or_missing_model() -> None:
    wrapper = BugModelWrapper()
    wrapper.is_loaded = True

    with pytest.raises(RuntimeError, match=r"Bug prediction model is not loaded\. Call load\(\) first"):
        wrapper.predict_class({})


@pytest.mark.parametrize(
    ("raw_probability", "expected_prediction", "expected_probability", "expected_defective"),
    [
        (0.49994, 0, 0.4999, False),
        (0.5, 1, 0.5, True),
        (1.4, 1, 1.0, True),
        (-0.2, 0, 0.0, False),
    ],
)
def test_bug_predict_maps_threshold_rounding_and_probability_bounds(
    raw_probability: float,
    expected_prediction: int,
    expected_probability: float,
    expected_defective: bool,
) -> None:
    wrapper = BugModelWrapper()
    wrapper.is_loaded = True
    wrapper.feature_schema = ["cbo", "wmc"]
    wrapper.model = Mock()
    booster = wrapper.model.get_booster.return_value
    booster.predict.return_value = np.array([raw_probability])
    dataframe = object()
    dmatrix = object()
    metrics = {"cbo": 2, "wmc": 5.0}

    with (
        patch("app.models.bug_model_wrapper.build_bug_feature_dataframe", return_value=dataframe) as build,
        patch("app.models.bug_model_wrapper.xgb.DMatrix", return_value=dmatrix) as dmatrix_type,
    ):
        result = wrapper.predict_class(metrics)

    build.assert_called_once_with(metrics)
    dmatrix_type.assert_called_once_with(dataframe, feature_names=["cbo", "wmc"])
    booster.predict.assert_called_once_with(dmatrix)
    assert result == {
        "prediction": expected_prediction,
        "probability_score": expected_probability,
        "is_defective": expected_defective,
    }


def test_bug_predict_propagates_invalid_input_and_booster_failures() -> None:
    wrapper = BugModelWrapper()
    wrapper.is_loaded = True
    wrapper.model = Mock()

    with (
        patch(
            "app.models.bug_model_wrapper.build_bug_feature_dataframe",
            side_effect=TypeError("metrics must be a mapping"),
        ),
        pytest.raises(TypeError, match="metrics must be a mapping"),
    ):
        wrapper.predict_class(None)

    with (
        patch("app.models.bug_model_wrapper.build_bug_feature_dataframe", return_value=object()),
        patch("app.models.bug_model_wrapper.xgb.DMatrix", return_value=object()),
    ):
        wrapper.model.get_booster.return_value.predict.side_effect = RuntimeError("booster failed")
        with pytest.raises(RuntimeError, match="booster failed"):
            wrapper.predict_class({"cbo": 1})
