import json
from pathlib import Path
import xgboost as xgb
import joblib

def predict(comments: list[str]):
    base_dir = Path(__file__).resolve().parent
    
    # 1. Load Vectorizer
    vectorizer = joblib.load(str(base_dir / "tfidf_vectorizer.joblib"))
    
    # 2. Load XGBoost model & classes
    model = xgb.XGBClassifier()
    model.load_model(str(base_dir / "tfidf_xgboost_model.json"))
    
    with open(base_dir / "classes.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    id2label = meta["id2label"]

    # 3. Transform features
    X_feats = vectorizer.transform(comments)
    
    # 4. Predict
    preds = model.predict(X_feats)
    probs = model.predict_proba(X_feats)
    
    results = []
    for c, pred_id, prob in zip(comments, preds, probs):
        label = id2label[str(pred_id)]
        confidence = float(prob[pred_id])
        results.append({
            "comment": c,
            "category": label,
            "confidence": round(confidence, 4),
            "is_debt": label != "non_debt"
        })
    return results

if __name__ == "__main__":
    sample_comments = [
        "TODO: refactor this method to use the new caching layer",
        "FIXME: this causes a null pointer exception under heavy load",
        "Standard getter method for user ID",
        "Need to add unit tests for edge cases in this validator",
        "Requires API version 2.0 to support pagination"
    ]
    predictions = predict(sample_comments)
    for p in predictions:
        print(f"Comment: {p['comment']}")
        print(f"  -> Prediction: {p['category']} (Confidence: {p['confidence'] * 100:.1f}%, Debt: {p['is_debt']})\n")
