from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from features import build_feature_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved churn model and write figures/reports.")
    parser.add_argument("--data-path", type=str, default="data/raw/telco_churn.csv")
    parser.add_argument("--artifact-dir", type=str, default="artifacts")
    parser.add_argument("--report-dir", type=str, default="reports")
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data_path)
    fa = build_feature_frame(df)
    X, y = fa.X, fa.y

    # Mirror the split used in training for a simple first pass.
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=args.random_state
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=args.random_state
    )

    model = joblib.load(artifact_dir / "model.pkl")
    with open(artifact_dir / "model_metadata.json", "r", encoding="utf-8") as f:
        metadata = json.load(f)
    threshold = float(metadata["threshold"])

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    # ROC
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_score, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(figure_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    # PR
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(figure_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    # Confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title(f"Confusion Matrix @ threshold={threshold:.3f}")
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    scored = X_test.copy()
    scored["churn_probability"] = y_score
    scored["predicted_churn"] = y_pred
    if "MonthlyCharges" in X_test.columns:
        scored["estimated_monthly_revenue_at_risk"] = X_test["MonthlyCharges"].values * y_score
    scored.to_csv(report_dir / "test_set_scored.csv", index=False)

    total_high_risk = int((y_pred == 1).sum())
    total_at_risk = float(scored.get("estimated_monthly_revenue_at_risk", pd.Series(dtype=float)).sum())

    summary = f"""# Executive Summary

## Model
- Champion model: {metadata['champion_model']}
- Classification threshold: {threshold:.3f}

## Test Set Snapshot
- Customers scored: {len(scored)}
- Predicted high-risk customers: {total_high_risk}
- Estimated monthly revenue at risk: ${total_at_risk:,.2f}

## Notes
- This first-pass report uses the saved champion pipeline and the held-out test split.
- Revenue at risk is a proxy based on churn probability × monthly charges.
- Next iteration: add calibration plots, driver analysis, and monitoring/drift reporting.
"""

    with open(report_dir / "executive_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)

    print(f"Wrote figures to {figure_dir.resolve()}")
    print(f"Wrote summary to {(report_dir / 'executive_summary.md').resolve()}")


if __name__ == "__main__":
    main()
