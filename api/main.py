from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from prometheus_fastapi_instrumentator import Instrumentator
from pydantic import BaseModel

from src.features import engineer_features

ARTIFACT_DIR = Path("artifacts")
MODEL = joblib.load(ARTIFACT_DIR / "model.pkl") if (ARTIFACT_DIR / "model.pkl").exists() else None
METADATA = json.loads((ARTIFACT_DIR / "model_metadata.json").read_text()) if (ARTIFACT_DIR / "model_metadata.json").exists() else None
API_KEY = os.getenv("API_KEY")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

# Patch older pickled models that may miss attributes after version bumps
def _patch_model(model):
    try:
        lr = model.named_steps.get("model") if hasattr(model, "named_steps") else None
        if lr and lr.__class__.__name__ == "LogisticRegression" and not hasattr(lr, "multi_class"):
            lr.multi_class = "auto"
    except Exception:
        pass
    return model

app = FastAPI(title="Customer Churn Scoring API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Expose /metrics for Prometheus scraping (must be added before startup)
Instrumentator().instrument(app).expose(app)


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


def verify_api_key(api_key: Optional[str] = Depends(api_key_header)) -> Optional[str]:
    if API_KEY and api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
            headers={"WWW-Authenticate": "API-Key"},
        )
    return api_key


def risk_tier(prob: float) -> str:
    if prob < 0.30:
        return "low"
    if prob < 0.60:
        return "medium"
    return "high"


def recommended_action(prob: float, threshold: float, monthly_charges: Optional[float] = None) -> str:
    """Align API action playbook with batch scoring logic.

    The top tier still respects the model threshold while adding more
    granular, telco-friendly categories so downstream dashboards and API
    clients stay consistent.
    """

    # Top tier: well above threshold – treat as save attempt
    if prob >= max(threshold + 0.15, 0.80):
        return "Save offer: retention specialist call + bill credit"

    # At/above threshold: personalized outreach
    if prob >= threshold:
        return "High-touch outreach: contract renewal or loyalty perks"

    # Mid tier: likely churn but below threshold – try lower-cost nudge
    if prob >= 0.55:
        return "Proactive nudge: speed/bundle upgrade email"

    # Low-mid tier: watchlist with engagement
    if prob >= 0.33:
        return "Monitor + CSAT survey follow-up"

    # Optional slight prioritization for high-value customers even if low risk
    if monthly_charges and monthly_charges >= 120:
        return "Monitor + CSAT survey follow-up"

    return "No intervention needed"


def _predict_dicts(records: List[Dict[str, Any]]) -> List[PredictionResponse]:
    if MODEL is None or METADATA is None:
        raise RuntimeError("Model artifacts not found. Train the model first.")

    model = _patch_model(MODEL)

    df = pd.DataFrame(records)
    features = engineer_features(df)
    drop_cols = [c for c in ["customerID", "Churn", "ChurnFlag"] if c in features.columns]
    X = features.drop(columns=drop_cols)
    scores = model.predict_proba(X)[:, 1]
    threshold = float(METADATA["threshold"])

    outputs = []
    for record, prob in zip(records, scores):
        outputs.append(PredictionResponse(
            customer_id=record.get("customerID"),
            churn_probability=float(prob),
            predicted_churn=int(prob >= threshold),
            risk_tier=risk_tier(float(prob)),
            recommended_action=recommended_action(float(prob), threshold, record.get("MonthlyCharges")),
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
def predict(payload: CustomerPayload, _: Optional[str] = Depends(verify_api_key)) -> PredictionResponse:
    return _predict_dicts([payload.model_dump()])[0]


@app.post("/batch-predict", response_model=List[PredictionResponse])
def batch_predict(payloads: List[CustomerPayload], _: Optional[str] = Depends(verify_api_key)) -> List[PredictionResponse]:
    return _predict_dicts([p.model_dump() for p in payloads])
