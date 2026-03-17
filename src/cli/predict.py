from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    logger.warning("SHAP unavailable: %s", e)
    SHAP_AVAILABLE = False

from src.features import build_feature_frame
from src.kkbox_features import build_kkbox_feature_frame
from src.utils.io import ensure_dir, load_config, read_table
from src.utils.themes import map_feature_to_theme


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--input", type=str, required=True)
    return parser.parse_args()


def estimate_monthly_value(row: pd.Series, dataset_type: str) -> float:
    """Heuristic monthly revenue proxy.

    Telco: use MonthlyCharges when available (fallback: TotalCharges / tenure).
    KKBox/other: use payment-derived signals as before.
    """

    if dataset_type == "telco":
        # Primary: explicit monthly charge
        charge = row.get("MonthlyCharges")
        if charge is not None and not pd.isna(charge):
            try:
                return float(charge)
            except Exception:
                pass

        # Fallback: total divided by tenure months
        total = row.get("TotalCharges")
        tenure = row.get("tenure")
        if total is not None and tenure not in (None, 0, "0", "0.0"):
            try:
                return float(total) / float(tenure)
            except Exception:
                return 0.0

        return 0.0

    # KKBox & others: use payment history
    candidates = [
        ("latest_actual_amount_paid", "latest_payment_plan_days"),
        ("avg_amount_paid", None),
        ("amount_paid_per_txn", None),
        ("total_amount_paid", None),
    ]

    for amount_col, days_col in candidates:
        amount = row.get(amount_col)
        if amount is None or pd.isna(amount):
            continue

        if days_col:
            days = row.get(days_col)
            if days is not None and not pd.isna(days) and days > 0:
                return float(amount) * 30.0 / float(days)
        else:
            return float(amount)

    return 0.0


def assign_risk_tier(prob: float, threshold: float) -> str:
    if prob >= max(threshold, 0.66):
        return "High"
    if prob >= 0.33:
        return "Medium"
    return "Low"


def recommended_action(prob: float, threshold: float) -> str:
    """Return a richer, telco-friendly action playbook string.

    We use more granularity than the previous three-bucket version so the
    dashboard's action summary has multiple, meaningful categories to show.
    Threshold is still respected for the top tier so business rules stay
    intact.
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

    return "No intervention needed"


def build_row_level_explanations(model, df: pd.DataFrame, top_n: int = 3):
    if not SHAP_AVAILABLE:
        return pd.DataFrame(index=df.index)

    try:
        preprocessor = model.named_steps["preprocessor"]
        estimator = model.named_steps["model"]

        if not hasattr(estimator, "feature_importances_"):
            return pd.DataFrame(index=df.index)

        X_transformed = preprocessor.transform(df)
        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        rows = []
        for row_idx in range(shap_values.shape[0]):
            vals = shap_values[row_idx]
            order = np.argsort(np.abs(vals))[::-1][:top_n]

            feature_list = []
            theme_scores = {}

            for idx in order:
                feat = str(feature_names[idx])
                contribution = float(vals[idx])
                direction = "increases" if contribution > 0 else "decreases"
                theme = map_feature_to_theme(feat)

                feature_list.append(f"{feat} ({direction} risk)")
                theme_scores[theme] = theme_scores.get(theme, 0.0) + abs(contribution)

            top_theme = max(theme_scores, key=theme_scores.get) if theme_scores else "Other"

            rows.append(
                {
                    "top_driver_1": feature_list[0] if len(feature_list) > 0 else None,
                    "top_driver_2": feature_list[1] if len(feature_list) > 1 else None,
                    "top_driver_3": feature_list[2] if len(feature_list) > 2 else None,
                    "primary_driver_theme": top_theme,
                }
            )

        return pd.DataFrame(rows, index=df.index)

    except Exception as e:
        logger.exception("Could not build row-level explanations: %s", e)
        return pd.DataFrame(index=df.index)


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    cfg = load_config(args.config)

    artifact_dir = Path(cfg["artifacts"]["dir"])
    dataset_type = cfg["data"].get("dataset_type", "telco")

    # Keep telco outputs in data/predictions, route others (e.g., kkbox) to subdirectories
    prediction_dir = ensure_dir(Path("data/predictions") / (dataset_type if dataset_type != "telco" else ""))

    model = joblib.load(artifact_dir / "model.pkl")

    with open(artifact_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)

    threshold = float(metadata.get("threshold", 0.5))

    raw_df = read_table(args.input)
    target_col = cfg["data"].get("target_col", "Churn" if dataset_type == "telco" else "is_churn")

    if dataset_type == "kkbox":
        feature_artifacts = build_kkbox_feature_frame(raw_df, target_col=target_col)
    else:
        feature_artifacts = build_feature_frame(raw_df)
    X = feature_artifacts.X

    probs = model.predict_proba(X)[:, 1]

    out = raw_df.copy()
    out["churn_probability"] = probs
    out["predicted_churn"] = (out["churn_probability"] >= threshold).astype(int)
    out["risk_tier"] = out["churn_probability"].apply(lambda x: assign_risk_tier(x, threshold))
    out["recommended_action"] = out["churn_probability"].apply(lambda x: recommended_action(x, threshold))
    out["model_threshold"] = threshold
    out["scored_at"] = datetime.utcnow().isoformat()
    out["estimated_monthly_value"] = out.apply(lambda row: estimate_monthly_value(row, dataset_type), axis=1)
    out["expected_revenue_at_risk"] = out["churn_probability"] * out["estimated_monthly_value"]

    explanation_df = build_row_level_explanations(model, X)
    if not explanation_df.empty:
        out = pd.concat([out, explanation_df], axis=1)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = prediction_dir / f"predictions_{timestamp}.csv"
    latest_path = prediction_dir / "latest_predictions.csv"

    out.to_csv(output_path, index=False)
    out.to_csv(latest_path, index=False)

    logger.info("Saved predictions to %s", output_path.resolve())
    logger.info("Updated latest predictions at %s", latest_path.resolve())
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Prediction run failed.")
        sys.exit(1)
