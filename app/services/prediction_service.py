"""Main ML orchestration service for SATD classification and Bug Prediction."""

from typing import Any
import logging

from app.models.satd_model_wrapper import SATDModelWrapper
from app.models.bug_model_wrapper import BugModelWrapper
from app.schemas.prediction_request import ClassMetricInput, PredictionJobRequest, CommentItem
from app.schemas.prediction_response import (
    SATDDetectionResponse,
    BugPredictionResponse,
    ClassPredictionResponse,
    PredictionJobResponse,
)

logger = logging.getLogger(__name__)


class PredictionService:
    """Orchestrates model inference across both SATD and Bug Prediction models."""

    def __init__(self) -> None:
        self.satd_model = SATDModelWrapper()
        self.bug_model = BugModelWrapper()
        self._is_initialized = False

    def load_models(self) -> None:
        """Load both models on service startup."""
        if not self._is_initialized:
            logger.info("Loading SATD classification model...")
            self.satd_model.load()

            logger.info("Loading Bug Prediction model...")
            self.bug_model.load()

            self._is_initialized = True
            logger.info("All ML models successfully loaded into memory.")

    def is_ready(self) -> bool:
        """Check if all models are loaded and ready for inference."""
        return self.satd_model.is_loaded and self.bug_model.is_loaded

    def predict_satd_comment(self, comment_text: str, comment_id: int | None = None) -> SATDDetectionResponse:
        """Classify a single comment for SATD."""
        if not self._is_initialized:
            self.load_models()

        res = self.satd_model.predict_comment(comment_text)
        return SATDDetectionResponse(
            commentId=comment_id,
            comment=res["comment"],
            category=res["category"],
            confidenceScore=res["confidence_score"],
            isDebt=res["is_debt"]
        )

    def predict_class(self, class_input: ClassMetricInput) -> ClassPredictionResponse:
        """Perform bug prediction and comment SATD detection for a single class."""
        if not self._is_initialized:
            self.load_models()

        # 1. Bug prediction on the 28 numerical metrics
        metrics_dict = class_input.model_dump(by_alias=False)
        bug_res = self.bug_model.predict_class(metrics_dict)
        bug_prediction = BugPredictionResponse(
            prediction=bug_res["prediction"],
            probabilityScore=bug_res["probability_score"],
            isDefective=bug_res["is_defective"]
        )

        # 2. SATD prediction on each extracted comment
        satd_detections: list[SATDDetectionResponse] = []
        for item in class_input.comments:
            if isinstance(item, CommentItem):
                text = item.comment
                c_id = item.comment_id
            elif isinstance(item, dict):
                text = item.get("comment", "")
                c_id = item.get("commentId") or item.get("comment_id")
            else:
                text = str(item)
                c_id = None

            if text and text.strip():
                satd_detections.append(self.predict_satd_comment(text, c_id))

        return ClassPredictionResponse(
            classId=class_input.class_id,
            className=class_input.class_name,
            filePath=class_input.file_path,
            startLine=class_input.start_line,
            endLine=class_input.end_line,
            bugPrediction=bug_prediction,
            satdDetections=satd_detections
        )

    def predict_job(self, job_request: PredictionJobRequest) -> PredictionJobResponse:
        """Process an entire analysis job batch."""
        if not self._is_initialized:
            self.load_models()

        class_responses: list[ClassPredictionResponse] = []
        defective_count = 0
        total_comments = 0
        total_satd = 0

        for class_item in job_request.classes:
            class_res = self.predict_class(class_item)
            class_responses.append(class_res)

            if class_res.bug_prediction.is_defective:
                defective_count += 1

            for satd in class_res.satd_detections:
                total_comments += 1
                if satd.is_debt:
                    total_satd += 1

        return PredictionJobResponse(
            jobId=job_request.job_id,
            repositoryId=job_request.repository_id,
            status="SUCCESS",
            totalClassesAnalyzed=len(class_responses),
            defectiveClassesCount=defective_count,
            totalCommentsClassified=total_comments,
            totalSatdCount=total_satd,
            classes=class_responses
        )


# Global singleton instance
prediction_service = PredictionService()