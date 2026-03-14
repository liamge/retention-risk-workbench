from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay
from sklearn.model_selection import train_test_split

from src.features import build_feature_frame
from src.kkbox_features import build_kkbox_feature_frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved churn model and write figures/reports.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--data-path", type=str, default=None, help="Optional override for data path")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Optional override for artifact dir")
    parser.add_argument("--report-dir", type=str, default="reports")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def time_based_split(X, y, cfg: Dict) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    train_end = pd.to_datetime(cfg["split"]["train_end"])
    dev_end = pd.to_datetime(cfg["split"]["dev_end"])

    if "transaction_date" not in X.columns:
        raise ValueError("transaction_date column required for time-based split.")

    tx = pd.to_datetime(X["transaction_date"], errors="coerce")

    train_mask = tx <= train_end
    dev_mask = (tx > train_end) & (tx <= dev_end)
    test_mask = tx > dev_end

    if not train_mask.any() or not dev_mask.any() or not test_mask.any():
        raise ValueError("Time split produced empty partitions; adjust train_end/dev_end in config.")

    X_train, y_train = X[train_mask], y[train_mask]
    X_dev, y_dev = X[dev_mask], y[dev_mask]
    X_test, y_test = X[test_mask], y[test_mask]
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    data_path = args.data_path or cfg["data"]["raw_path"]
    dataset_type = cfg["data"].get("dataset_type", "telco")
    target_col = cfg["data"].get("target_col", "Churn" if dataset_type == "telco" else "is_churn")
    artifact_dir = Path(args.artifact_dir or cfg["artifacts"]["dir"])
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = report_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(data_path)
    if dataset_type == "kkbox":
        fa = build_kkbox_feature_frame(df, target_col=target_col)
        splitter = time_based_split
    else:
        fa = build_feature_frame(df)
        splitter = lambda X, y, cfg: train_test_split(  # type: ignore
            X, y, test_size=0.30, stratify=y, random_state=cfg["split"].get("random_state", 42)
        )

    X, y = fa.X, fa.y

    # Mirror the split used in training for a simple first pass.
    X_train, X_dev, X_test, y_train, y_dev, y_test = splitter(X, y, cfg)

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
