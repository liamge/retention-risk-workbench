from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

try:
    import shap
    SHAP_AVAILABLE = True
except Exception as e:
    print(f"SHAP unavailable: {e}")
    SHAP_AVAILABLE = False

from src.features import build_feature_frame
from src.kkbox_features import build_kkbox_feature_frame


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--input", type=str, required=True)
    return parser.parse_args()


def load_config(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input data file not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input file type for {path}. Use CSV or Parquet.")


def estimate_monthly_value(row: pd.Series) -> float:
    """Heuristic monthly revenue proxy that works for KKBox engineered features."""
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
    if prob >= threshold:
        return "Priority retention outreach"
    if prob >= 0.33:
        return "Monitor and light-touch engagement"
    return "No intervention needed"


def map_feature_to_theme(feature_name: str) -> str:
    """Group feature names into business-friendly themes (supports KKBox + Telco)."""
    name = feature_name.lower()

    # KKBox engagement & listening behavior
    if any(k in name for k in ["songs_played", "total_secs", "completion_rate", "skip_rate", "near_completion", "quality_score", "repeat_ratio", "log_day", "secs_per_unique", "avg_song"]):
        return "Engagement & Listening"
    # KKBox subscription / billing / payments
    if any(k in name for k in ["payment_method", "payment_plan", "plan_list_price", "actual_amount_paid", "amount_paid", "auto_renew", "cancel", "paid_to_list", "amount_paid_per_txn"]):
        return "Subscription & Billing"
    # Renewal / expiry risk
    if any(k in name for k in ["membership_expire", "post_expiry", "early_renewal", "latest_cancel", "latest_auto_renew"]):
        return "Renewal & Expiry"
    # Recency of activity and transactions
    if any(k in name for k in ["days_since_last", "latest_log", "latest_transaction", "last_log_date", "last_transaction_date"]):
        return "Recency & Activity"
    # Content breadth / diversity
    if any(k in name for k in ["num_unq", "secs_per_unique"]):
        return "Content Breadth"
    # Customer profile & registration
    if any(k in name for k in ["city", "age", "gender", "registered_via", "registration_init", "account_age"]):
        return "Customer Profile"

    # Telco legacy themes
    if any(k in name for k in ["contract", "paperless", "payment", "month_to_month", "electronic_check", "auto_pay"]):
        return "Billing & Contract"
    if any(k in name for k in ["tenure", "customer_stage", "tenure_group"]):
        return "Customer Lifecycle"
    if any(k in name for k in ["monthlycharges", "totalcharges", "avg_monthly_spend", "price_ratio", "high_bill", "revenue"]):
        return "Pricing & Value"
    if any(k in name for k in ["techsupport", "onlinesecurity", "onlinebackup", "deviceprotection", "support"]):
        return "Support & Protection"
    if any(k in name for k in ["internetservice", "fiber", "streaming", "phoneservice", "multipleservices", "num_services", "service"]):
        return "Product Usage"
    if any(k in name for k in ["senior", "partner", "dependents", "gender"]):
        return "Customer Profile"

    return "Other"


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
        print(f"Could not build row-level explanations: {e}")
        return pd.DataFrame(index=df.index)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    artifact_dir = Path(cfg["artifacts"]["dir"])
    dataset_type = cfg["data"].get("dataset_type", "telco")

    # Keep telco outputs in data/predictions, route others (e.g., kkbox) to subdirectories
    prediction_dir = Path("data/predictions") / (dataset_type if dataset_type != "telco" else "")
    prediction_dir.mkdir(parents=True, exist_ok=True)

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
    out["estimated_monthly_value"] = out.apply(estimate_monthly_value, axis=1)
    out["expected_revenue_at_risk"] = out["churn_probability"] * out["estimated_monthly_value"]

    explanation_df = build_row_level_explanations(model, X)
    if not explanation_df.empty:
        out = pd.concat([out, explanation_df], axis=1)

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = prediction_dir / f"predictions_{timestamp}.csv"
    latest_path = prediction_dir / "latest_predictions.csv"

    out.to_csv(output_path, index=False)
    out.to_csv(latest_path, index=False)

    print(f"Saved predictions to {output_path}")
    print(f"Updated latest predictions at {latest_path}")


if __name__ == "__main__":
    main()
