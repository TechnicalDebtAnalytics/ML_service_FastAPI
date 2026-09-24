"""Isolated unit tests for the SATD and bug-prediction model wrappers."""

import json
from unittest.mock import Mock, call, patch

import numpy as np
import pytest

from app.config.settings import settings
from app.models.bug_model_wrapper import BugModelWrapper
from app.models.satd_model_wrapper import SATDModelWrapper
from app.preprocessing.bug_preprocessor import CANONICAL_FEATURES


SATD_FILENAMES = [
    settings.SATD_SVM_MODEL_FILE,
    settings.SATD_WORD_TFIDF_FILE,
    settings.SATD_CHAR_TFIDF_FILE,
    settings.SATD_LABEL_ENCODER_FILE,
    settings.SATD_FEATURES_FILE,
]


def test_satd_load_reads_all_artifacts_and_marks_wrapper_loaded(tmp_path) -> None:
    for filename in SATD_FILENAMES:
        (tmp_path / filename).touch()
    artifacts = [Mock(name="model"), Mock(name="word"), Mock(name="char"), Mock(name="encoder"), {"DESIGN": {}}]
    wrapper = SATDModelWrapper(tmp_path)

    with patch("app.models.satd_model_wrapper.joblib.load", side_effect=artifacts) as load:
        wrapper.load()

    assert load.call_args_list == [call(tmp_path / filename) for filename in SATD_FILENAMES]
    assert wrapper.model is artifacts[0]
    assert wrapper.word_vectorizer is artifacts[1]
    assert wrapper.char_vectorizer is artifacts[2]
    assert wrapper.encoder is artifacts[3]
    assert wrapper.satd_features == artifacts[4]
    assert wrapper.is_loaded is True


def test_satd_load_rejects_first_missing_artifact_without_loading_anything(tmp_path) -> None:
    wrapper = SATDModelWrapper(tmp_path)

    with (
        patch("app.models.satd_model_wrapper.joblib.load") as load,
        pytest.raises(FileNotFoundError, match="SATD model artifact not found"),
    ):
        wrapper.load()

    load.assert_not_called()
    assert wrapper.is_loaded is False


def test_satd_load_propagates_corrupt_artifact_failure_and_stays_unloaded(tmp_path) -> None:
    for filename in SATD_FILENAMES:
        (tmp_path / filename).touch()
    wrapper = SATDModelWrapper(tmp_path)

    with (
        patch("app.models.satd_model_wrapper.joblib.load", side_effect=ValueError("corrupt pickle")),
        pytest.raises(ValueError, match="corrupt pickle"),
    ):
        wrapper.load()

    assert wrapper.is_loaded is False


def test_satd_predict_rejects_unloaded_model() -> None:
    with pytest.raises(RuntimeError, match=r"SATD model is not loaded\. Call load\(\) first"):
        SATDModelWrapper().predict_comment("// TODO")


def test_satd_predict_maps_multiclass_label_confidence_and_features() -> None:
    wrapper = loaded_satd_wrapper()
    features = object()
    wrapper.model.predict.return_value = np.array([2])
    wrapper.model.decision_function.return_value = np.array([[1.0, 3.0]])
    wrapper.encoder.inverse_transform.return_value = np.array(["DESIGN_DEBT"])

    with (
        patch("app.models.satd_model_wrapper.clean_comment", return_value="todo refactor") as clean,
        patch("app.models.satd_model_wrapper.build_satd_feature_vector", return_value=features) as build,
    ):
        result = wrapper.predict_comment("// TODO refactor")

    clean.assert_called_once_with("// TODO refactor")
    build.assert_called_once_with(
        "todo refactor", wrapper.word_vectorizer, wrapper.char_vectorizer, wrapper.satd_features
    )
    wrapper.model.predict.assert_called_once_with(features)
    wrapper.encoder.inverse_transform.assert_called_once_with(np.array([2]))
    assert result == {
        "comment": "// TODO refactor",
        "category": "DESIGN_DEBT",
        "confidence_score": 0.8808,
        "is_debt": True,
    }


@pytest.mark.parametrize("label", ["WITHOUT_CLASSIFICATION", "non_debt", "Clean"])
def test_satd_predict_maps_non_debt_labels_case_insensitively(label: str) -> None:
    wrapper = loaded_satd_wrapper()
    wrapper.model.predict.return_value = np.array([0])
    wrapper.model.decision_function.return_value = np.array([0.0])
    wrapper.encoder.inverse_transform.return_value = np.array([label])

    with (
        patch("app.models.satd_model_wrapper.clean_comment", return_value="ordinary comment"),
        patch("app.models.satd_model_wrapper.build_satd_feature_vector", return_value=object()),
    ):
        result = wrapper.predict_comment("ordinary comment")

    assert result["category"] == label
    assert result["confidence_score"] == 0.5
    assert result["is_debt"] is False


def test_satd_predict_returns_default_without_inference_for_empty_or_none_input() -> None:
    wrapper = loaded_satd_wrapper()

    assert wrapper.predict_comment(None) == {
        "comment": None,
        "category": "WITHOUT_CLASSIFICATION",
        "confidence_score": 1.0,
        "is_debt": False,
    }
    wrapper.model.predict.assert_not_called()
    wrapper.encoder.inverse_transform.assert_not_called()


def test_satd_predict_uses_fallback_confidence_when_decision_function_fails() -> None:
    wrapper = loaded_satd_wrapper()
    wrapper.model.predict.return_value = np.array([1])
    wrapper.model.decision_function.side_effect = RuntimeError("decision unavailable")
    wrapper.encoder.inverse_transform.return_value = np.array(["CODE_DEBT"])

    with (
        patch("app.models.satd_model_wrapper.clean_comment", return_value="fix later"),
        patch("app.models.satd_model_wrapper.build_satd_feature_vector", return_value=object()),
    ):
        result = wrapper.predict_comment("fix later")

    assert result["confidence_score"] == 0.85
    assert result["is_debt"] is True


def test_satd_predict_propagates_feature_or_model_failure() -> None:
    wrapper = loaded_satd_wrapper()

    with (
        patch("app.models.satd_model_wrapper.clean_comment", return_value="todo"),
        patch(
            "app.models.satd_model_wrapper.build_satd_feature_vector",
            side_effect=ValueError("invalid feature shape"),
        ),
        pytest.raises(ValueError, match="invalid feature shape"),
    ):
        wrapper.predict_comment("TODO")


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


def loaded_satd_wrapper() -> SATDModelWrapper:
    wrapper = SATDModelWrapper()
    wrapper.is_loaded = True
    wrapper.model = Mock()
    wrapper.word_vectorizer = Mock()
    wrapper.char_vectorizer = Mock()
    wrapper.encoder = Mock()
    wrapper.satd_features = {"DESIGN": {"todo": 1.0}}
    return wrapper
