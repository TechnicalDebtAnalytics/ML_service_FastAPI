"""Wrapper and loader for the XGBoost Bug Prediction classification model."""

import json
from pathlib import Path
from typing import Any
import xgboost as xgb

from app.config.settings import settings
from app.preprocessing.bug_preprocessor import build_bug_feature_dataframe, CANONICAL_FEATURES


class BugModelWrapper:
    """Manages Bug Prediction model loading and inference."""

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or settings.BUG_MODEL_DIR
        self.model: xgb.XGBClassifier | None = None
        self.feature_schema: list[str] = CANONICAL_FEATURES
        self.is_loaded: bool = False

    def load(self) -> None:
        """Load trained XGBoost model and schema."""
        model_path = self.model_dir / settings.BUG_MODEL_FILE
        schema_path = self.model_dir / settings.BUG_SCHEMA_FILE

        if not model_path.exists():
            raise FileNotFoundError(f"Bug prediction model artifact not found: {model_path}")

        if schema_path.exists():
            try:
                schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
                if isinstance(schema_data, list):
                    self.feature_schema = schema_data
                elif isinstance(schema_data, dict) and "features" in schema_data:
                    self.feature_schema = schema_data["features"]
            except Exception:
                self.feature_schema = CANONICAL_FEATURES

        classifier = xgb.XGBClassifier()
        classifier.load_model(str(model_path))
        self.model = classifier
        self.is_loaded = True

    def predict_class(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Predict bug defect risk for a class metric vector."""
        if not self.is_loaded or self.model is None:
            raise RuntimeError("Bug prediction model is not loaded. Call load() first.")

        df = build_bug_feature_dataframe(metrics)
        dmatrix = xgb.DMatrix(df, feature_names=self.feature_schema)
        booster = self.model.get_booster()

        defect_prob = float(booster.predict(dmatrix)[0])
        prediction = 1 if defect_prob >= 0.5 else 0
        bounded_prob = max(0.0, min(1.0, round(defect_prob, 4)))

        return {
            "prediction": prediction,
            "probability_score": bounded_prob,
            "is_defective": prediction == 1
        }
