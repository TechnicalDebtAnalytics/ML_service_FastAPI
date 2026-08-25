"""Health check and model readiness endpoints."""

from fastapi import APIRouter
from app.config.settings import settings
from app.services.prediction_service import prediction_service

router = APIRouter(tags=["Health"])


@router.get("/health")
def health_check() -> dict[str, object]:
    """Return health status and model availability."""
    satd_loaded = prediction_service.satd_model.is_loaded
    bug_loaded = prediction_service.bug_model.is_loaded
    overall_ready = satd_loaded and bug_loaded

    return {
        "status": "UP" if overall_ready else "DEGRADED",
        "app_name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "models": {
            "satd_model_loaded": satd_loaded,
            "bug_model_loaded": bug_loaded,
        },
        "rabbitmq": {
            "enabled": settings.RABBITMQ_ENABLED,
            "host": settings.RABBITMQ_HOST,
            "job_queue": settings.ML_JOB_QUEUE,
            "result_queue": settings.ML_RESULT_QUEUE,
        }
    }
