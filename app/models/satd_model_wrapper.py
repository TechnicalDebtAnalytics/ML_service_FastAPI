"""Wrapper and loader for the SATD LinearSVC classification model."""

import os
from pathlib import Path
from typing import Any
import joblib
import numpy as np

from app.config.settings import settings
from app.preprocessing.satd_preprocessor import clean_comment, build_satd_feature_vector


class SATDModelWrapper:
    """Manages SATD model loading and inference."""

    def __init__(self, model_dir: Path | None = None):
        self.model_dir = model_dir or settings.SATD_MODEL_DIR
        self.model: Any = None
        self.word_vectorizer: Any = None
        self.char_vectorizer: Any = None
        self.encoder: Any = None
        self.satd_features: dict[str, dict[str, float]] = {}
        self.is_loaded: bool = False

    def load(self) -> None:
        """Load model artifacts into memory."""
        svm_path = self.model_dir / settings.SATD_SVM_MODEL_FILE
        word_tfidf_path = self.model_dir / settings.SATD_WORD_TFIDF_FILE
        char_tfidf_path = self.model_dir / settings.SATD_CHAR_TFIDF_FILE
        encoder_path = self.model_dir / settings.SATD_LABEL_ENCODER_FILE
        features_path = self.model_dir / settings.SATD_FEATURES_FILE

        for path in (svm_path, word_tfidf_path, char_tfidf_path, encoder_path, features_path):
            if not path.exists():
                raise FileNotFoundError(f"SATD model artifact not found: {path}")

        self.model = joblib.load(svm_path)
        self.word_vectorizer = joblib.load(word_tfidf_path)
        self.char_vectorizer = joblib.load(char_tfidf_path)
        self.encoder = joblib.load(encoder_path)
        self.satd_features = joblib.load(features_path)
        self.is_loaded = True

    def predict_comment(self, comment: str) -> dict[str, Any]:
        """Classify a single comment and return predicted category and normalized confidence."""
        if not self.is_loaded:
            raise RuntimeError("SATD model is not loaded. Call load() first.")

        cleaned = clean_comment(comment)
        if not cleaned:
            return {
                "comment": comment,
                "category": "WITHOUT_CLASSIFICATION",
                "confidence_score": 1.0,
                "is_debt": False
            }

        features = build_satd_feature_vector(
            cleaned,
            self.word_vectorizer,
            self.char_vectorizer,
            self.satd_features
        )

        prediction = self.model.predict(features)
        label = str(self.encoder.inverse_transform(prediction)[0])

        # Compute calibrated confidence score bounded in [0.0, 1.0] for DB constraint
        try:
            decision = self.model.decision_function(features)
            if hasattr(decision, "ndim") and decision.ndim > 1:
                # Softmax across classes for multi-class decision values
                exp_vals = np.exp(decision[0] - np.max(decision[0]))
                probs = exp_vals / np.sum(exp_vals)
                confidence = float(np.max(probs))
            else:
                # Sigmoid for binary decision values
                raw_val = float(decision[0]) if hasattr(decision, "__getitem__") else float(decision)
                confidence = float(1.0 / (1.0 + np.exp(-raw_val)))
        except Exception:
            confidence = 0.85

        confidence = max(0.0, min(1.0, round(confidence, 4)))
        is_debt = label.upper() not in ("WITHOUT_CLASSIFICATION", "NON_DEBT", "CLEAN")

        return {
            "comment": comment,
            "category": label,
            "confidence_score": confidence,
            "is_debt": is_debt
        }
