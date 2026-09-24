"""Isolated unit tests for PredictionService orchestration."""

from unittest.mock import Mock, call, patch

import pytest

from app.schemas.prediction_request import ClassMetricInput, CommentItem, PredictionJobRequest
from app.schemas.prediction_response import (
    BugPredictionResponse,
    ClassPredictionResponse,
    SATDDetectionResponse,
)
from app.services.prediction_service import PredictionService


@pytest.fixture
def service() -> PredictionService:
    prediction_service = PredictionService()
    prediction_service.satd_model = Mock()
    prediction_service.bug_model = Mock()
    prediction_service.satd_model.is_loaded = False
    prediction_service.bug_model.is_loaded = False
    return prediction_service


def test_load_models_loads_each_wrapper_only_once(service: PredictionService) -> None:
    service.load_models()
    service.load_models()

    service.satd_model.load.assert_called_once_with()
    service.bug_model.load.assert_called_once_with()
    assert service._is_initialized is True


def test_load_models_stops_when_satd_model_loading_fails(service: PredictionService) -> None:
    service.satd_model.load.side_effect = RuntimeError("SATD artifacts unavailable")

    with pytest.raises(RuntimeError, match="SATD artifacts unavailable"):
        service.load_models()

    service.bug_model.load.assert_not_called()
    assert service._is_initialized is False


def test_load_models_remains_uninitialized_when_bug_model_loading_fails(service: PredictionService) -> None:
    service.bug_model.load.side_effect = RuntimeError("bug model unavailable")

    with pytest.raises(RuntimeError, match="bug model unavailable"):
        service.load_models()

    service.satd_model.load.assert_called_once_with()
    service.bug_model.load.assert_called_once_with()
    assert service._is_initialized is False


@pytest.mark.parametrize(
    ("satd_loaded", "bug_loaded", "expected"),
    [
        (False, False, False),
        (True, False, False),
        (False, True, False),
        (True, True, True),
    ],
)
def test_is_ready_requires_both_models(
    service: PredictionService,
    satd_loaded: bool,
    bug_loaded: bool,
    expected: bool,
) -> None:
    service.satd_model.is_loaded = satd_loaded
    service.bug_model.is_loaded = bug_loaded

    assert service.is_ready() is expected
    service.satd_model.load.assert_not_called()
    service.bug_model.load.assert_not_called()


def test_predict_satd_comment_maps_model_result_and_comment_id(service: PredictionService) -> None:
    service._is_initialized = True
    service.satd_model.predict_comment.return_value = {
        "comment": "// TODO remove workaround",
        "category": "DESIGN_DEBT",
        "confidence_score": 0.91,
        "is_debt": True,
    }

    response = service.predict_satd_comment("// TODO remove workaround", comment_id=17)

    service.satd_model.predict_comment.assert_called_once_with("// TODO remove workaround")
    assert response.comment_id == 17
    assert response.comment == "// TODO remove workaround"
    assert response.category == "DESIGN_DEBT"
    assert response.confidence_score == 0.91
    assert response.is_debt is True
    service.satd_model.load.assert_not_called()
    service.bug_model.load.assert_not_called()


def test_predict_satd_comment_lazily_loads_models(service: PredictionService) -> None:
    service.satd_model.predict_comment.return_value = {
        "comment": "plain comment",
        "category": "WITHOUT_CLASSIFICATION",
        "confidence_score": 1.0,
        "is_debt": False,
    }

    service.predict_satd_comment("plain comment")

    service.satd_model.load.assert_called_once_with()
    service.bug_model.load.assert_called_once_with()
    service.satd_model.predict_comment.assert_called_once_with("plain comment")
    assert service._is_initialized is True


def test_predict_satd_comment_propagates_inference_failure(service: PredictionService) -> None:
    service._is_initialized = True
    service.satd_model.predict_comment.side_effect = RuntimeError("SATD inference failed")

    with pytest.raises(RuntimeError, match="SATD inference failed"):
        service.predict_satd_comment("// TODO")


def test_predict_class_combines_bug_prediction_and_nonblank_satd_comments(
    service: PredictionService,
) -> None:
    service._is_initialized = True
    service.bug_model.predict_class.return_value = {
        "prediction": 1,
        "probability_score": 0.82,
        "is_defective": True,
    }
    service.satd_model.predict_comment.side_effect = [
        {
            "comment": "// TODO refactor",
            "category": "DESIGN_DEBT",
            "confidence_score": 0.9,
            "is_debt": True,
        },
        {
            "comment": "Useful explanation",
            "category": "WITHOUT_CLASSIFICATION",
            "confidence_score": 0.8,
            "is_debt": False,
        },
    ]
    class_input = ClassMetricInput(
        classId=501,
        className="DebtCalculator",
        filePath="src/DebtCalculator.java",
        startLine=10,
        endLine=80,
        fanIn=3,
        comments=[
            CommentItem(commentId=601, comment="// TODO refactor"),
            "   ",
            "Useful explanation",
        ],
    )

    response = service.predict_class(class_input)

    metrics = service.bug_model.predict_class.call_args.args[0]
    assert metrics["class_id"] == 501
    assert metrics["class_name"] == "DebtCalculator"
    assert metrics["fan_in"] == 3
    assert response.class_id == 501
    assert response.class_name == "DebtCalculator"
    assert response.bug_prediction.prediction == 1
    assert response.bug_prediction.probability_score == 0.82
    assert response.bug_prediction.is_defective is True
    assert [detection.comment_id for detection in response.satd_detections] == [601, None]
    assert [detection.is_debt for detection in response.satd_detections] == [True, False]
    assert service.satd_model.predict_comment.call_args_list == [
        call("// TODO refactor"),
        call("Useful explanation"),
    ]


def test_predict_class_lazily_loads_once_before_all_inference(service: PredictionService) -> None:
    service.bug_model.predict_class.return_value = {
        "prediction": 0,
        "probability_score": 0.1,
        "is_defective": False,
    }
    service.satd_model.predict_comment.return_value = {
        "comment": "first",
        "category": "WITHOUT_CLASSIFICATION",
        "confidence_score": 0.9,
        "is_debt": False,
    }

    service.predict_class(ClassMetricInput(comments=["first", "second"]))

    service.satd_model.load.assert_called_once_with()
    service.bug_model.load.assert_called_once_with()
    assert service.satd_model.predict_comment.call_count == 2


def test_predict_class_propagates_bug_inference_failure_without_running_satd(
    service: PredictionService,
) -> None:
    service._is_initialized = True
    service.bug_model.predict_class.side_effect = ValueError("invalid metric vector")

    with pytest.raises(ValueError, match="invalid metric vector"):
        service.predict_class(ClassMetricInput(comments=["// TODO"]))

    service.satd_model.predict_comment.assert_not_called()


def test_predict_job_aggregates_class_and_satd_results(service: PredictionService) -> None:
    service._is_initialized = True
    first_input = ClassMetricInput(classId=1, className="First")
    second_input = ClassMetricInput(classId=2, className="Second")
    first_response = class_response(
        class_id=1,
        class_name="First",
        is_defective=True,
        detections=[satd_response(11, True), satd_response(12, False)],
    )
    second_response = class_response(
        class_id=2,
        class_name="Second",
        is_defective=False,
        detections=[satd_response(21, True)],
    )

    with patch.object(service, "predict_class", side_effect=[first_response, second_response]) as predict_class:
        response = service.predict_job(
            PredictionJobRequest(jobId="job-100", repositoryId="repo-42", classes=[first_input, second_input])
        )

    assert predict_class.call_args_list == [call(first_input), call(second_input)]
    assert response.job_id == "job-100"
    assert response.repository_id == "repo-42"
    assert response.status == "SUCCESS"
    assert response.total_classes_analyzed == 2
    assert response.defective_classes_count == 1
    assert response.total_comments_classified == 3
    assert response.total_satd_count == 2
    assert response.classes == [first_response, second_response]


def test_predict_job_with_no_classes_returns_zero_totals_and_lazily_initializes(
    service: PredictionService,
) -> None:
    response = service.predict_job(PredictionJobRequest(jobId="empty-job", repositoryId="repo-42", classes=[]))

    service.satd_model.load.assert_called_once_with()
    service.bug_model.load.assert_called_once_with()
    service.satd_model.predict_comment.assert_not_called()
    service.bug_model.predict_class.assert_not_called()
    assert response.status == "SUCCESS"
    assert response.total_classes_analyzed == 0
    assert response.defective_classes_count == 0
    assert response.total_comments_classified == 0
    assert response.total_satd_count == 0
    assert response.classes == []


def test_predict_job_propagates_class_inference_failure_and_stops_processing(
    service: PredictionService,
) -> None:
    service._is_initialized = True
    class_inputs = [
        ClassMetricInput(classId=1, className="First"),
        ClassMetricInput(classId=2, className="Second"),
        ClassMetricInput(classId=3, className="NeverProcessed"),
    ]
    first_response = class_response(1, "First", False, [])

    with patch.object(
        service,
        "predict_class",
        side_effect=[first_response, RuntimeError("class inference failed")],
    ) as predict_class:
        with pytest.raises(RuntimeError, match="class inference failed"):
            service.predict_job(PredictionJobRequest(jobId="job-100", classes=class_inputs))

    assert predict_class.call_args_list == [call(class_inputs[0]), call(class_inputs[1])]


def satd_response(comment_id: int, is_debt: bool) -> SATDDetectionResponse:
    return SATDDetectionResponse(
        commentId=comment_id,
        comment=f"comment-{comment_id}",
        category="DESIGN_DEBT" if is_debt else "WITHOUT_CLASSIFICATION",
        confidenceScore=0.9,
        isDebt=is_debt,
    )


def class_response(
    class_id: int,
    class_name: str,
    is_defective: bool,
    detections: list[SATDDetectionResponse],
) -> ClassPredictionResponse:
    return ClassPredictionResponse(
        classId=class_id,
        className=class_name,
        filePath=f"src/{class_name}.java",
        startLine=1,
        endLine=10,
        bugPrediction=BugPredictionResponse(
            prediction=1 if is_defective else 0,
            probabilityScore=0.8 if is_defective else 0.2,
            isDefective=is_defective,
        ),
        satdDetections=detections,
    )
