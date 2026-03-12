from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

from src.features import engineer_features

ARTIFACT_DIR = Path("artifacts")
MODEL = joblib.load(ARTIFACT_DIR / "model.pkl") if (ARTIFACT_DIR / "model.pkl").exists() else None
METADATA = json.loads((ARTIFACT_DIR / "model_metadata.json").read_text()) if (ARTIFACT_DIR / "model_metadata.json").exists() else None

app = FastAPI(title="Customer Churn Scoring API", version="0.1.0")


class CustomerPayload(BaseModel):
    customerID: Optional[str] = None
    gender: Optional[str] = None
    SeniorCitizen: Optional[int] = None
    Partner: Optional[str] = None
    Dependents: Optional[str] = None
    tenure: Optional[int] = None
    PhoneService: Optional[str] = None
    MultipleLines: Optional[str] = None
    InternetService: Optional[str] = None
    OnlineSecurity: Optional[str] = None
    OnlineBackup: Optional[str] = None
    DeviceProtection: Optional[str] = None
    TechSupport: Optional[str] = None
    StreamingTV: Optional[str] = None
    StreamingMovies: Optional[str] = None
    Contract: Optional[str] = None
    PaperlessBilling: Optional[str] = None
    PaymentMethod: Optional[str] = None
    MonthlyCharges: Optional[float] = None
    TotalCharges: Optional[Any] = None


class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    churn_probability: float
    predicted_churn: int
    risk_tier: str
    recommended_action: str
    threshold: float


def risk_tier(prob: float) -> str:
    if prob < 0.30:
        return "low"
    if prob < 0.60:
        return "medium"
    return "high"


def recommended_action(prob: float, monthly_charges: Optional[float]) -> str:
    if prob >= 0.70:
        return "priority retention outreach"
    if prob >= 0.45:
        return "targeted save offer"
    if monthly_charges and monthly_charges >= 80:
        return "light-touch account review"
    return "monitor"


def _predict_dicts(records: List[Dict[str, Any]]) -> List[PredictionResponse]:
    if MODEL is None or METADATA is None:
        raise RuntimeError("Model artifacts not found. Train the model first.")

    df = pd.DataFrame(records)
    features = engineer_features(df)
    drop_cols = [c for c in ["customerID", "Churn", "ChurnFlag"] if c in features.columns]
    X = features.drop(columns=drop_cols)
    scores = MODEL.predict_proba(X)[:, 1]
    threshold = float(METADATA["threshold"])

    outputs = []
    for record, prob in zip(records, scores):
        outputs.append(PredictionResponse(
            customer_id=record.get("customerID"),
            churn_probability=float(prob),
            predicted_churn=int(prob >= threshold),
            risk_tier=risk_tier(float(prob)),
            recommended_action=recommended_action(float(prob), record.get("MonthlyCharges")),
            threshold=threshold,
        ))
    return outputs


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok" if MODEL is not None else "model_not_loaded"}


@app.get("/model-info")
def model_info() -> Dict[str, Any]:
    if METADATA is None:
        return {"status": "model_not_loaded"}
    return METADATA


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CustomerPayload) -> PredictionResponse:
    return _predict_dicts([payload.model_dump()])[0]


@app.post("/batch-predict", response_model=List[PredictionResponse])
def batch_predict(payloads: List[CustomerPayload]) -> List[PredictionResponse]:
    return _predict_dicts([p.model_dump() for p in payloads])
