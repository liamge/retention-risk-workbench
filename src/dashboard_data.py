from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import pandas as pd


def _safe_read_json(path: Path) -> Dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def load_artifacts(artifact_dir: Path) -> Dict:
    metadata = _safe_read_json(artifact_dir / "model_metadata.json")
    metrics = _safe_read_json(artifact_dir / "metrics.json")
    candidate_results = _safe_read_csv(artifact_dir / "candidate_model_results.csv")
    shap_importance = _safe_read_csv(artifact_dir / "figures" / "champion_test_shap_importance.csv")

    return {
        "metadata": metadata,
        "metrics": metrics,
        "candidate_results": candidate_results,
        "shap_importance": shap_importance,
    }


def load_latest_predictions(prediction_dir: Path) -> Optional[pd.DataFrame]:
    if not prediction_dir.exists():
        return None

    csvs = sorted(prediction_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return None

    return pd.read_csv(csvs[0])


def summarize_technical_view(metrics: Dict, metadata: Dict) -> Dict:
    test_metrics = metadata.get("test_metrics", {})
    if not test_metrics and "test" in metrics:
        test_metrics = metrics["test"]

    return {
        "roc_auc": test_metrics.get("roc_auc", 0.0),
        "pr_auc": test_metrics.get("pr_auc", 0.0),
        "f1": test_metrics.get("f1", 0.0),
        "precision": test_metrics.get("precision", 0.0),
        "recall": test_metrics.get("recall", 0.0),
        "threshold": metadata.get("threshold", test_metrics.get("threshold", 0.5)),
    }


def summarize_business_view(pred_df: Optional[pd.DataFrame], metadata: Dict) -> Dict:
    if pred_df is None or pred_df.empty:
        return {
            "high_risk_count": 0,
            "high_risk_rate": 0.0,
            "revenue_at_risk": 0.0,
            "action_summary": pd.DataFrame(),
            "theme_summary": pd.DataFrame(),
            "insights": ["No prediction batch available yet."],
        }

    threshold = metadata.get("threshold", 0.5)

    if "risk_tier" not in pred_df.columns:
        pred_df = pred_df.copy()
        pred_df["risk_tier"] = pd.cut(
            pred_df["churn_probability"],
            bins=[0, 0.33, 0.66, 1.0],
            labels=["Low", "Medium", "High"],
            include_lowest=True,
        )

    high_risk = pred_df[pred_df["churn_probability"] >= threshold].copy()

    monthly_col = "MonthlyCharges" if "MonthlyCharges" in pred_df.columns else None
    if monthly_col:
        revenue_at_risk = float((high_risk["churn_probability"] * high_risk[monthly_col]).sum())
    else:
        revenue_at_risk = 0.0

    action_summary = (
        high_risk["recommended_action"]
        .value_counts(dropna=False)
        .rename_axis("recommended_action")
        .reset_index(name="customer_count")
        if "recommended_action" in high_risk.columns and not high_risk.empty
        else pd.DataFrame(columns=["recommended_action", "customer_count"])
    )

    theme_summary = (
        high_risk["primary_driver_theme"]
        .value_counts(dropna=False)
        .rename_axis("driver_theme")
        .reset_index(name="customer_count")
        if "primary_driver_theme" in high_risk.columns and not high_risk.empty
        else pd.DataFrame(columns=["driver_theme", "customer_count"])
    )

    top_theme = theme_summary.iloc[0]["driver_theme"] if not theme_summary.empty else "N/A"

    insights = [
        f"{len(high_risk)} customers are above the current intervention threshold.",
        f"Estimated near-term revenue at risk is ${revenue_at_risk:,.0f}.",
        f"The most common driver theme among high-risk customers is {top_theme}.",
    ]

    return {
        "high_risk_count": int(len(high_risk)),
        "high_risk_rate": float(len(high_risk) / len(pred_df)),
        "revenue_at_risk": revenue_at_risk,
        "action_summary": action_summary,
        "theme_summary": theme_summary,
        "insights": insights,
    }


def top_risk_customers(pred_df: Optional[pd.DataFrame], top_n: int = 25) -> Optional[pd.DataFrame]:
    if pred_df is None or pred_df.empty:
        return pred_df

    cols = [
        c
        for c in [
            "customerID",
            "churn_probability",
            "risk_tier",
            "recommended_action",
            "primary_driver_theme",
            "top_driver_1",
            "top_driver_2",
            "top_driver_3",
            "MonthlyCharges",
            "tenure",
            "Contract",
        ]
        if c in pred_df.columns
    ]

    return pred_df.sort_values("churn_probability", ascending=False)[cols].head(top_n)