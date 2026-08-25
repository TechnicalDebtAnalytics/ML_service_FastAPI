"""FastAPI Application entry point."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import sys
from pathlib import Path

# Ensure project root is on sys.path regardless of working directory
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config.settings import settings
from app.services.prediction_service import prediction_service
from app.messaging.rabbitmq_consumer import rabbitmq_consumer
from app.api.routes import health, prediction

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s"
)
logger = logging.getLogger("debtlens.ml")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager: load models and manage background workers."""
    logger.info("Initializing %s v%s...", settings.APP_NAME, settings.APP_VERSION)

    # 1. Load ML models into memory
    try:
        prediction_service.load_models()
        logger.info("ML models ready.")
    except Exception as e:
        logger.error("Failed to load ML models during startup: %s", e, exc_info=True)

    # 2. Start RabbitMQ consumer thread if enabled
    if settings.RABBITMQ_ENABLED:
        try:
            rabbitmq_consumer.start_consuming()
        except Exception as e:
            logger.warning("RabbitMQ consumer failed to start: %s", e)

    yield

    # Shutdown logic
    logger.info("Shutting down %s...", settings.APP_NAME)
    if settings.RABBITMQ_ENABLED:
        rabbitmq_consumer.stop_consuming()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="DebtLens Machine Learning Backend for Bug Prediction and SATD Comment Classification",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Routers
app.include_router(health.router)
app.include_router(prediction.router)


@app.get("/", tags=["Root"])
def root() -> dict[str, str]:
    """Root endpoint providing service metadata and API documentation links."""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "status": "RUNNING"
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
