from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from src.utils.io import read_table
from src.utils.themes import map_feature_to_theme


def _safe_read_json(path: Path) -> Dict:
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def _safe_read_csv(path: Path) -> pd.DataFrame:
    if path.exists():
        return read_table(path)
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


def load_causal_artifacts(artifact_dir: Path) -> Dict:
    causal_dir = artifact_dir / "causal"
    summary = _safe_read_json(causal_dir / "summary.json")
    uplift_table = _safe_read_csv(causal_dir / "uplift_table.csv")
    cate = _safe_read_csv(causal_dir / "cate.csv")
    policy = _safe_read_csv(causal_dir / "policy_recommendations.csv")

    return {
        "summary": summary,
        "uplift_table": uplift_table,
        "cate": cate,
        "policy": policy,
    }


def load_latest_predictions(prediction_dir: Path) -> Optional[pd.DataFrame]:
    if not prediction_dir.exists():
        return None

    csvs = sorted(prediction_dir.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not csvs:
        return None

    return read_table(csvs[0])


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


def summarize_causal_view(causal_artifacts: Dict) -> Dict:
    summary = causal_artifacts.get("summary", {}) or {}
    uplift_table = causal_artifacts.get("uplift_table", pd.DataFrame())
    policy = causal_artifacts.get("policy", pd.DataFrame())

    cumulative_uplift = float(uplift_table.get("cumulative_uplift", pd.Series(dtype=float)).iloc[-1]) if not uplift_table.empty else 0.0
    top_bin_uplift = float(uplift_table.get("uplift", pd.Series(dtype=float)).iloc[0]) if not uplift_table.empty else 0.0

    insights = []
    if summary:
        ate_pct = summary.get("ate", 0.0) * 100
        direction = "reduced" if summary.get("ate", 0) < 0 else "increased"
        insights.append(f"Treatment {direction} churn by {abs(ate_pct):.1f} pts on average.")
        if not pd.isna(summary.get("uplift_top_bin", np.nan)):
            insights.append(
                f"Top uplift bin shows {summary.get('uplift_top_bin', 0.0)*100:.1f} pt lift; focus top {summary.get('budget_fraction', 0.0)*100:.0f}% customers."
            )
    if policy is not None and not policy.empty:
        total_gain = float(policy["expected_gain"].sum()) if "expected_gain" in policy.columns else 0.0
        insights.append(f"Expected net gain from policy: {total_gain:,.1f} value units.")

    return {
        "ate": float(summary.get("ate", 0.0)),
        "treated_mean": float(summary.get("treated_mean", 0.0)),
        "control_mean": float(summary.get("control_mean", 0.0)),
        "uplift_direction": summary.get("uplift_direction", "unknown"),
        "qini": float(summary.get("qini", cumulative_uplift)),
        "top_bin_uplift": float(summary.get("uplift_top_bin", top_bin_uplift)),
        "budget_fraction": float(summary.get("budget_fraction", 0.0)),
        "uplift_table": uplift_table,
        "policy": policy,
        "cate": causal_artifacts.get("cate", pd.DataFrame()),
        "insights": insights,
    }


def _theme_summary_from_shap(metadata: Dict) -> pd.DataFrame:
    shap_source = metadata.get("shap_importance_df")
    if shap_source is None or shap_source.empty:
        return pd.DataFrame(columns=["driver_theme", "importance"])

    shap_source = shap_source.copy()
    shap_source["driver_theme"] = shap_source["feature"].apply(map_feature_to_theme)
    theme_summary = (
        shap_source.groupby("driver_theme")[["mean_abs_shap"]]
        .sum()
        .rename(columns={"mean_abs_shap": "importance"})
        .reset_index()
        .sort_values("importance", ascending=False)
    )
    return theme_summary


def summarize_business_view(pred_df: Optional[pd.DataFrame], metadata: Dict) -> Dict:
    if pred_df is None or pred_df.empty:
        theme_summary = _theme_summary_from_shap(metadata)
        return {
            "high_risk_count": 0,
            "high_risk_rate": 0.0,
            "revenue_at_risk": 0.0,
            "action_summary": pd.DataFrame(),
            "theme_summary": theme_summary,
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

    value_col = None
    for cand in ["estimated_monthly_value", "MonthlyCharges", "avg_amount_paid", "latest_actual_amount_paid", "amount_paid_per_txn"]:
        if cand in pred_df.columns:
            value_col = cand
            break

    if value_col:
        base_values = pred_df[value_col].fillna(0).clip(lower=0)
        revenue_at_risk = float((high_risk["churn_probability"] * high_risk[value_col].fillna(0).clip(lower=0)).sum())
        expected_loss_total = float((pred_df["churn_probability"] * base_values).sum())
        avg_value_high_risk = float(high_risk[value_col].mean()) if not high_risk.empty else 0.0
    else:
        revenue_at_risk = 0.0
        expected_loss_total = 0.0
        avg_value_high_risk = 0.0

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

    # Fallback to global SHAP importance if row-level driver themes are missing
    if theme_summary.empty:
        theme_summary = _theme_summary_from_shap(metadata)

    # Ensure string-backed columns avoid Arrow LargeUtf8 issues in Streamlit
    for df in (action_summary, theme_summary):
        for col in df.columns:
            if df[col].dtype == "string[python]" or df[col].dtype == "string":
                df[col] = df[col].astype("string[python]")
            elif df[col].dtype == object:
                df[col] = df[col].astype("string[python]")

    top_theme = theme_summary.iloc[0]["driver_theme"] if not theme_summary.empty else "N/A"

    insights = [
        f"{len(high_risk)} customers are above the current intervention threshold.",
        f"Estimated near-term revenue at risk (high-risk only): ${revenue_at_risk:,.0f}.",
        f"Portfolio-wide expected churn loss: ${expected_loss_total:,.0f} per month.",
        f"Average value of a high-risk customer: ${avg_value_high_risk:,.0f}.",
        f"The most common driver theme among high-risk customers is {top_theme}.",
    ]

    return {
        "high_risk_count": int(len(high_risk)),
        "high_risk_rate": float(len(high_risk) / len(pred_df)),
        "revenue_at_risk": revenue_at_risk,
        "expected_loss_total": expected_loss_total,
        "avg_value_high_risk": avg_value_high_risk,
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
