"""REST API routes for synchronous predictions, debugging, and testing."""

from fastapi import APIRouter, HTTPException, status
from app.schemas.prediction_request import (
    ClassMetricInput,
    PredictionJobRequest,
    SATDCommentRequest,
)
from app.schemas.prediction_response import (
    ClassPredictionResponse,
    PredictionJobResponse,
    SATDDetectionResponse,
)
from app.services.prediction_service import prediction_service

router = APIRouter(prefix="/api/v1/predict", tags=["Predictions"])


@router.post("/satd", response_model=SATDDetectionResponse, status_code=status.HTTP_200_OK)
def predict_satd_comment(request: SATDCommentRequest) -> SATDDetectionResponse:
    """Classify a single source code comment for Self-Admitted Technical Debt (SATD)."""
    try:
        return prediction_service.predict_satd_comment(request.comment)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"SATD classification failed: {str(e)}"
        )


@router.post("/class", response_model=ClassPredictionResponse, status_code=status.HTTP_200_OK)
def predict_class_metrics(class_input: ClassMetricInput) -> ClassPredictionResponse:
    """Predict bug defect probability and classify comments for a single class."""
    try:
        return prediction_service.predict_class(class_input)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Class prediction failed: {str(e)}"
        )


@router.post("/job", response_model=PredictionJobResponse, status_code=status.HTTP_200_OK)
def predict_analysis_job(job_request: PredictionJobRequest) -> PredictionJobResponse:
    """Batch execute bug and SATD predictions for all classes in an analysis job."""
    try:
        return prediction_service.predict_job(job_request)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis job prediction failed: {str(e)}"
        )