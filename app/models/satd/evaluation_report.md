# TF-IDF + XGBoost (Approach B) Final Model Evaluation Report

**Model:** XGBoost Classifier on 5,000 TF-IDF Unigrams & Bigrams  
**Dataset:** `satd-dataset-code_comments.csv` (62,275 comments across 10 projects)  
**Evaluation Protocol:** 5-Fold Grouped Cross-Validation (by project)  
**Trained On:** 100% Full Dataset with Balanced Sample Weights  
**Date:** 2026-09-26 18:00:54  

---

## 1. Executive Summary

| Metric | Cross-Validation Score |
|---|---|
| **Mean Macro-F1** | **0.4349 ± 0.0624** |
| **Mean Accuracy** | **94.22%** |
| **Mean Weighted-F1** | **0.9468** |
| **Full Train Time** | **50.78 s** |

---

## 2. 5-Fold Cross-Validation Breakdown

| Fold | Macro-F1 | Accuracy | Training Time |
|:---:|:---:|:---:|:---:|
| Fold 1 | 0.4540 | 96.54% | 38.5 s |
| Fold 2 | 0.4557 | 90.84% | 32.0 s |
| Fold 3 | 0.5042 | 95.45% | 37.5 s |
| Fold 4 | 0.3175 | 94.67% | 38.9 s |
| Fold 5 | 0.4429 | 93.59% | 44.1 s |
| **Mean ± Std** | **0.4349 ± 0.0624** | **94.22%** | **38.2 s** |

## 3. Out-of-Fold Per-Class Performance

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| `code/design_debt` | 0.4895 | 0.6238 | 0.5486 | 2703 |
| `defect_debt` | 0.2311 | 0.2394 | 0.2352 | 472 |
| `documentation_debt` | 0.5000 | 0.0741 | 0.1290 | 54 |
| `non_debt` | 0.9892 | 0.9698 | 0.9794 | 58204 |
| `requirement_debt` | 0.3250 | 0.4980 | 0.3933 | 757 |
| `test_debt` | 0.2768 | 0.3647 | 0.3147 | 85 |

---

## 4. Hyperparameters & Configuration

```python
TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    sublinear_tf=True
)

xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="mlogloss",
    random_state=42,
    tree_method="hist"
)
```

---

## 5. Artifact Files in `final_model/`

- `tfidf_vectorizer.joblib`: Trained TF-IDF feature extractor (5k n-gram features).
- `tfidf_xgboost_model.json`: Native XGBoost model file (fast inference & multi-platform compatibility).
- `tfidf_xgboost_model.joblib`: Scikit-learn XGBClassifier joblib serialization.
- `label_encoder.joblib`: Trained scikit-learn LabelEncoder.
- `classes.json`: Class ID to name mappings.
- `training_report.json`: Machine-readable metrics & confusion matrix.
- `evaluation_report.md`: Human-readable evaluation summary.
- `inference_example.py`: Python script demonstrating inference on new text comments.
