from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.dashboard_data import (
    load_artifacts,
    load_causal_artifacts,
    load_latest_predictions,
    summarize_business_view,
    summarize_causal_view,
    summarize_technical_view,
    top_risk_customers,
)

st.set_page_config(page_title="Customer Churn Intelligence Dashboard", layout="wide")

ROOT_ARTIFACT_DIR = Path("artifacts")
ROOT_PREDICTION_DIR = Path("data/predictions")


def _discover_artifact_options(root_dir: Path) -> dict[str, Path]:
    options = {"telco (artifacts/)": root_dir}
    if root_dir.exists():
        for sub in sorted(root_dir.iterdir()):
            if sub.is_dir() and (sub / "model_metadata.json").exists():
                options[f"{sub.name} (artifacts/{sub.name})"] = sub
    return options


artifact_options = _discover_artifact_options(ROOT_ARTIFACT_DIR)
selected_label = st.sidebar.selectbox("Choose dataset artifacts", list(artifact_options.keys()))
ARTIFACT_DIR = artifact_options[selected_label]

# Align predictions to dataset folder when present
pred_subdir = ARTIFACT_DIR.name if ARTIFACT_DIR != ROOT_ARTIFACT_DIR else ""
PREDICTION_DIR = ROOT_PREDICTION_DIR / pred_subdir if pred_subdir else ROOT_PREDICTION_DIR

artifacts = load_artifacts(ARTIFACT_DIR)
causal_artifacts = load_causal_artifacts(ARTIFACT_DIR)
pred_df = load_latest_predictions(PREDICTION_DIR)

metadata = artifacts["metadata"]
metrics = artifacts["metrics"]
candidate_results = artifacts["candidate_results"]
shap_importance = artifacts["shap_importance"]

# expose SHAP table for fallback driver theme aggregation
metadata["shap_importance_df"] = shap_importance

business = summarize_business_view(pred_df, metadata)
technical = summarize_technical_view(metrics, metadata)
causal = summarize_causal_view(causal_artifacts)

st.title("Customer Churn Intelligence Dashboard")
st.caption("Refresh after a new training or scoring run to load the latest artifacts.")
st.caption(f"Artifacts: {ARTIFACT_DIR} | Predictions: {PREDICTION_DIR}")

tab_exec, tab_tech, tab_models, tab_customers, tab_causal = st.tabs(
    [
        "Executive Summary",
        "Technical Performance",
        "Model Comparison",
        "High-Risk Customers",
        "Causal Uplift",
    ]
)

is_kkbox = "kkbox" in ARTIFACT_DIR.name.lower()

with tab_exec:
    st.subheader("Executive Summary")

    if is_kkbox:
        # KKBox-specific overview
        avg_prob = float(pred_df["churn_probability"].mean()) if pred_df is not None and "churn_probability" in pred_df.columns else 0.0
        top_theme = (
            business["theme_summary"].iloc[0]["driver_theme"]
            if not business["theme_summary"].empty
            else "N/A"
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("High-Risk Users", business["high_risk_count"])
        c2.metric("High-Risk Rate", f"{business['high_risk_rate']:.1%}")
        c3.metric("Revenue at Risk", f"${business['revenue_at_risk']:,.0f}")
        c4.metric("Expected Monthly Loss", f"${business.get('expected_loss_total', 0):,.0f}")

        st.markdown("**KKBox Risk Story**")
        st.write(
            "- High-risk users are those above the model threshold; focus outreach here.\n"
            "- Revenue metrics reflect expected loss using recent payment behavior.\n"
            f"- Top driver theme currently: **{top_theme}**. Broaden outreach playbooks accordingly.\n"
            "- Use the Technical and Model tabs to inspect curves and SHAP details."
        )
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Champion Model", metadata.get("champion_model", "N/A"))
        c2.metric("High-Risk Customers", business["high_risk_count"])
        c3.metric("High-Risk Rate", f"{business['high_risk_rate']:.1%}")
        c4.metric("Revenue at Risk", f"${business['revenue_at_risk']:,.0f}")

    st.markdown("### Key Insights")
    for insight in business["insights"]:
        st.write(f"- {insight}")

    left, right = st.columns(2)

    with left:
        st.markdown("### Recommended Actions")
        if not business["action_summary"].empty:
            st.dataframe(business["action_summary"], use_container_width=True)
            st.bar_chart(
                business["action_summary"].set_index("recommended_action")["customer_count"]
            )
        else:
            st.info("No action summary available yet.")

    with right:
        st.markdown("### Dominant Driver Themes")
        if not business["theme_summary"].empty:
            st.dataframe(business["theme_summary"], use_container_width=True)
            bar_source = (
                business["theme_summary"].set_index("driver_theme")["importance"]
                if "importance" in business["theme_summary"].columns
                else business["theme_summary"].set_index("driver_theme")["customer_count"]
            )
            st.bar_chart(bar_source)
        else:
            st.info("No driver theme summary available yet.")

    if pred_df is not None and not pred_df.empty:
        st.markdown("### Risk Distribution")
        risk_series = (
            pred_df["risk_tier"].value_counts().sort_index()
            if "risk_tier" in pred_df.columns
            else pd.cut(
                pred_df["churn_probability"],
                bins=[0, 0.33, 0.66, 1.0],
                labels=["Low", "Medium", "High"],
                include_lowest=True,
            ).value_counts().sort_index()
        )
        st.bar_chart(risk_series)

with tab_tech:
    st.subheader("Technical Performance")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ROC-AUC", f"{technical.get('roc_auc', 0):.3f}")
    c2.metric("PR-AUC", f"{technical.get('pr_auc', 0):.3f}")
    c3.metric("F1", f"{technical.get('f1', 0):.3f}")
    c4.metric("Recall", f"{technical.get('recall', 0):.3f}")
    c5.metric("Threshold", f"{technical.get('threshold', 0):.3f}")

    fig_dir = ARTIFACT_DIR / "figures"

    st.markdown("### Evaluation Artifacts")
    figure_map = {
        "ROC Curve": fig_dir / "champion_test_roc_curve.png",
        "Precision-Recall Curve": fig_dir / "champion_test_pr_curve.png",
        "Calibration Curve": fig_dir / "champion_test_calibration_curve.png",
        "Confusion Matrix": fig_dir / "champion_test_confusion_matrix.png",
        "Feature Importance": fig_dir / "champion_test_feature_importance.png",
        "SHAP Global Importance": fig_dir / "champion_test_shap_summary.png",
    }

    cols = st.columns(2)
    i = 0
    for label, path in figure_map.items():
        if path.exists():
            with cols[i % 2]:
                st.markdown(f"**{label}**")
                # use_column_width is supported across older Streamlit versions used on Streamlit Cloud
                st.image(str(path), use_column_width=True)
        i += 1

    st.markdown("### Top SHAP Features")
    if not shap_importance.empty:
        st.dataframe(shap_importance.head(20), use_container_width=True)
        st.bar_chart(shap_importance.head(15).set_index("feature")["mean_abs_shap"])
    else:
        st.info("No SHAP importance table found.")

    st.markdown("### Model Metadata")
    st.json(metadata)

with tab_models:
    st.subheader("Candidate Model Comparison")
    if candidate_results is not None and not candidate_results.empty:
        st.dataframe(candidate_results, use_container_width=True)

        metric_choice = st.selectbox(
            "Compare by metric",
            ["roc_auc", "pr_auc", "f1", "recall", "precision", "positive_rate"],
            index=0,
        )

        if metric_choice in candidate_results.columns:
            chart_df = candidate_results.set_index("model_name")[[metric_choice]]
            st.bar_chart(chart_df)
    else:
        st.info("No candidate model results found.")

with tab_customers:
    st.subheader("Highest-Risk Customers")
    high_risk_df = top_risk_customers(pred_df)

    if high_risk_df is not None and not high_risk_df.empty:
        st.dataframe(high_risk_df, use_container_width=True)

        if "primary_driver_theme" in high_risk_df.columns:
            st.markdown("### Driver Theme Mix in Top-Risk Customers")
            st.bar_chart(high_risk_df["primary_driver_theme"].value_counts())
    else:
        st.info("No scored prediction file found yet.")

with tab_causal:
    st.subheader("Causal Uplift & Policy")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("ATE (treatment - control)", f"{causal['ate']:.3f}")
    c2.metric("Top Bin Uplift", f"{causal['top_bin_uplift']:.3f}")
    c3.metric("Qini / Cum. Uplift", f"{causal['qini']:.3f}")
    c4.metric("Budget Fraction", f"{causal['budget_fraction']:.0%}")

    if causal.get("insights"):
        st.markdown("### Executive Takeaways")
        for insight in causal["insights"]:
            st.write(f"- {insight}")

    st.markdown("### Uplift Curve")
    uplift_df = causal.get("uplift_table")
    if uplift_df is not None and not uplift_df.empty:
        chart_df = uplift_df[["bin", "cumulative_uplift"]].set_index("bin")
        st.line_chart(chart_df)
        st.dataframe(uplift_df, use_container_width=True)
    else:
        st.info("No uplift table found. Generate causal artifacts with `src/reporting/causal_report.py`.")

    st.markdown("### Targeting Policy")
    policy_df = causal.get("policy")
    if policy_df is not None and not policy_df.empty:
        st.dataframe(policy_df, use_container_width=True)
    else:
        st.info("No policy recommendations available yet.")

    cate_df = causal.get("cate")
    if cate_df is not None and not cate_df.empty:
        st.markdown("### Segment CATE (largest absolute uplift first)")
        st.dataframe(cate_df.head(10), use_container_width=True)
