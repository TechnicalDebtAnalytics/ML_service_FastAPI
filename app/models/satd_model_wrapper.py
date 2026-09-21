"""Wrapper and loader for the SATD Transformer (CodeBERT / RoBERTa) classification model."""

from pathlib import Path
from typing import Any
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from app.config.settings import settings


class SATDModelWrapper:
    """Manages SATD CodeBERT transformer model loading and inference."""

    def __init__(self, model_dir: Path | str | None = None):
        self.model_dir = Path(model_dir) if model_dir else settings.SATD_MODEL_DIR
        self.tokenizer: Any = None
        self.model: Any = None
        self.device: torch.device | None = None
        self.id2label: dict[int | str, str] = {}
        self.is_loaded: bool = False

    def load(self) -> None:
        """Load tokenizer and model artifacts into memory."""
        model_path = str(self.model_dir)
        if not (self.model_dir / "config.json").exists():
            raise FileNotFoundError(f"SATD model config not found in: {self.model_dir}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()

        if hasattr(self.model.config, "id2label") and self.model.config.id2label:
            self.id2label = self.model.config.id2label
        else:
            self.id2label = {
                0: "non_debt",
                1: "code/design_debt",
                2: "requirement_debt",
                3: "defect_debt",
                4: "test_debt",
                5: "documentation_debt",
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
        batch_size: int = 32,
    ) -> list[dict[str, Any]]:
        """Classify a batch of comments and return predictions."""
        if not self.is_loaded:
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
                inputs = self.tokenizer(
                    valid_texts,
                    padding=True,
                    truncation=True,
                    max_length=512,
                    return_tensors="pt",
                ).to(self.device)

                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()

                for orig_idx, comment_text, prob_dist in zip(valid_indices, valid_texts, probs):
                    pred_id = int(prob_dist.argmax())
                    label = self.id2label.get(str(pred_id), self.id2label.get(pred_id, f"LABEL_{pred_id}"))
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
