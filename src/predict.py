from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import joblib
import pandas as pd

from src.features import engineer_features


RISK_BANDS = {
    "low": 0.30,
    "medium": 0.60,
}


def risk_tier(prob: float) -> str:
    if prob < RISK_BANDS["low"]:
        return "low"
    if prob < RISK_BANDS["medium"]:
        return "medium"
    return "high"


def recommended_action(prob: float, monthly_charges: float | None = None) -> str:
    if prob >= 0.70:
        return "priority retention outreach"
    if prob >= 0.45:
        return "targeted save offer"
    if monthly_charges and monthly_charges >= 80:
        return "light-touch account review"
    return "monitor"


def score_records(records: List[Dict], artifact_dir: str = "artifacts") -> List[Dict]:
    artifact_path = Path(artifact_dir)
    model = joblib.load(artifact_path / "model.pkl")
    with open(artifact_path / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    raw = pd.DataFrame(records)
    features = engineer_features(raw)
    drop_cols = [c for c in ["customerID", "Churn", "ChurnFlag"] if c in features.columns]
    X = features.drop(columns=drop_cols)
    scores = model.predict_proba(X)[:, 1]

    outputs = []
    for rec, prob in zip(records, scores):
        monthly_charges = rec.get("MonthlyCharges")
        outputs.append({
            "customer_id": rec.get("customerID"),
            "churn_probability": float(prob),
            "predicted_churn": int(prob >= metadata["threshold"]),
            "risk_tier": risk_tier(float(prob)),
            "recommended_action": recommended_action(float(prob), monthly_charges),
            "threshold": float(metadata["threshold"]),
        })
    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score customer records with the saved churn model.")
    parser.add_argument("--input", type=str, required=True, help="Path to CSV input data")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--output", type=str, default="predictions.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.input)
    outputs = score_records(df.to_dict(orient="records"), artifact_dir=args.artifact_dir)
    out_df = pd.DataFrame(outputs)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote predictions to {Path(args.output).resolve()}")


if __name__ == "__main__":
    main()
