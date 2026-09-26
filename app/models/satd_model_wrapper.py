"""Wrapper and loader for the SATD TF-IDF + XGBoost classification model."""

import json
from pathlib import Path
from typing import Any
import joblib
import xgboost as xgb

from app.config.settings import settings


class SATDModelWrapper:
    """Manages SATD TF-IDF feature extraction and XGBoost model inference."""

    def __init__(self, model_dir: Path | str | None = None):
        self.model_dir = Path(model_dir) if model_dir else settings.SATD_MODEL_DIR
        self.vectorizer: Any = None
        self.model: xgb.XGBClassifier | None = None
        self.id2label: dict[int | str, str] = {}
        self.is_loaded: bool = False

    def load(self) -> None:
        """Load TF-IDF vectorizer, XGBoost model, and class mappings into memory."""
        vectorizer_path = self.model_dir / settings.SATD_VECTORIZER_FILE
        model_path = self.model_dir / settings.SATD_MODEL_FILE
        classes_path = self.model_dir / settings.SATD_CLASSES_FILE

        if not vectorizer_path.exists():
            raise FileNotFoundError(f"SATD TF-IDF vectorizer not found: {vectorizer_path}")
        if not model_path.exists():
            raise FileNotFoundError(f"SATD XGBoost model artifact not found: {model_path}")

        # 1. Load TF-IDF Vectorizer
        self.vectorizer = joblib.load(str(vectorizer_path))

        # 2. Load XGBoost Model
        classifier = xgb.XGBClassifier()
        classifier.load_model(str(model_path))
        self.model = classifier

        # 3. Load Class Mappings
        if classes_path.exists():
            try:
                classes_data = json.loads(classes_path.read_text(encoding="utf-8"))
                if "id2label" in classes_data:
                    self.id2label = {int(k): str(v) for k, v in classes_data["id2label"].items()}
                elif "classes" in classes_data:
                    self.id2label = {int(i): str(cls_name) for i, cls_name in enumerate(classes_data["classes"])}
            except Exception:
                pass

        if not self.id2label:
            self.id2label = {
                0: "code/design_debt",
                1: "defect_debt",
                2: "documentation_debt",
                3: "non_debt",
                4: "requirement_debt",
                5: "test_debt",
            }

        self.is_loaded = True

    def predict_comment(self, comment: str) -> dict[str, Any]:
        """Classify a single comment and return predicted category and confidence."""
        results = self.predict_comments([comment])
        return results[0] if results else {
            "comment": comment,
            "category": "non_debt",
            "confidence_score": 1.0,
            "is_debt": False,
        }

    def predict_comments(
        self,
        comments: list[str],
        batch_size: int = 256,
    ) -> list[dict[str, Any]]:
        """Classify a batch of comments and return predictions."""
        if not self.is_loaded or self.model is None or self.vectorizer is None:
            raise RuntimeError("SATD model is not loaded. Call load() first.")

        if not comments:
            return []

        results: list[dict[str, Any]] = []

        for i in range(0, len(comments), batch_size):
            batch_texts = comments[i : i + batch_size]
            valid_indices = []
            valid_texts = []

            for idx, text in enumerate(batch_texts):
                if text is not None and str(text).strip():
                    valid_indices.append(idx)
                    valid_texts.append(str(text))

            batch_results: list[dict[str, Any] | None] = [None] * len(batch_texts)

            for idx, text in enumerate(batch_texts):
                if idx not in valid_indices:
                    batch_results[idx] = {
                        "comment": str(text) if text is not None else "",
                        "category": "non_debt",
                        "confidence_score": 1.0,
                        "is_debt": False,
                    }

            if valid_texts:
                # 1. Transform text with TF-IDF vectorizer
                tfidf_features = self.vectorizer.transform(valid_texts)

                # 2. Predict probability distribution
                probs = self.model.predict_proba(tfidf_features)

                for orig_idx, comment_text, prob_dist in zip(valid_indices, valid_texts, probs):
                    pred_id = int(prob_dist.argmax())
                    label = self.id2label.get(pred_id, self.id2label.get(str(pred_id), f"LABEL_{pred_id}"))
                    confidence = float(prob_dist[pred_id])
                    confidence = max(0.0, min(1.0, round(confidence, 4)))

                    is_debt = label.lower() not in ("without_classification", "non_debt", "clean")

                    batch_results[orig_idx] = {
                        "comment": comment_text,
                        "category": label,
                        "confidence_score": confidence,
                        "is_debt": is_debt,
                    }

            results.extend([res for res in batch_results if res is not None])

        return results
