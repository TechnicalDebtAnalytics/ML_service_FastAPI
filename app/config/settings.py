"""Application configuration settings."""

import os
from pathlib import Path
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# Load .env file into os.environ if present
_env_path = PROJECT_ROOT / ".env"
if _env_path.exists():
    try:
        with open(_env_path, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _k, _v = _line.split("=", 1)
                _k = _k.strip()
                _v = _v.strip().strip("'\"")
                if _k not in os.environ:
                    os.environ[_k] = _v
    except Exception:
        pass


class Settings(BaseModel):
    """Global configuration settings for ML FastAPI backend."""

    # API Configuration
    APP_NAME: str = "DebtLens ML Service"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = Field(default_factory=lambda: os.getenv("DEBUG", "false").lower() in ("true", "1"))

    # Model Directories
    SATD_MODEL_DIR: Path = BASE_DIR / "models" / "satd"
    BUG_MODEL_DIR: Path = BASE_DIR / "models" / "bug_prediction"

    # SATD Model Artifacts
    SATD_SVM_MODEL_FILE: str = "svm_satd_model.pkl"
    SATD_WORD_TFIDF_FILE: str = "word_tfidf.pkl"
    SATD_CHAR_TFIDF_FILE: str = "char_tfidf.pkl"
    SATD_LABEL_ENCODER_FILE: str = "label_encoder.pkl"
    SATD_FEATURES_FILE: str = "satd_features.pkl"

    # Bug Prediction Model Artifacts
    BUG_MODEL_FILE: str = "xgboost_bug_prediction.json"
    BUG_SCHEMA_FILE: str = "feature_schema.json"
    BUG_METADATA_FILE: str = "model_metadata.json"

    # RabbitMQ Configuration
    RABBITMQ_HOST: str = Field(default_factory=lambda: os.getenv("RABBITMQ_HOST", "localhost"))
    RABBITMQ_PORT: int = Field(default_factory=lambda: int(os.getenv("RABBITMQ_PORT", "5672")))
    RABBITMQ_USERNAME: str = Field(default_factory=lambda: os.getenv("RABBITMQ_USERNAME", "guest"))
    RABBITMQ_PASSWORD: str = Field(default_factory=lambda: os.getenv("RABBITMQ_PASSWORD", "guest"))
    RABBITMQ_VHOST: str = Field(default_factory=lambda: os.getenv("RABBITMQ_VHOST", "/"))
    RABBITMQ_ENABLED: bool = Field(default_factory=lambda: os.getenv("RABBITMQ_ENABLED", "true").lower() in ("true", "1"))

    # Exact Queue Names created in RabbitMQ
    ML_JOB_QUEUE: str = Field(default_factory=lambda: os.getenv("ML_JOB_QUEUE", "ML_job_cretion.queue"))
    ML_RESULT_QUEUE: str = Field(default_factory=lambda: os.getenv("ML_RESULT_QUEUE", "ML_job_results.queue"))


settings = Settings()