from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import shap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from src.utils.io import ensure_dir

logger = logging.getLogger(__name__)


def evaluate_scores(y_true: pd.Series, y_score: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_score >= threshold).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "threshold": float(threshold),
        "positive_rate": float(np.mean(y_pred)),
    }


def save_roc_curve(y_true: pd.Series, y_score: np.ndarray, outpath: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = roc_auc_score(y_true, y_score)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_pr_curve(y_true: pd.Series, y_score: np.ndarray, outpath: Path) -> None:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = average_precision_score(y_true, y_score)

    plt.figure(figsize=(6, 4))
    plt.plot(recall, precision, label=f"AP = {pr_auc:.3f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_calibration_curve(y_true: pd.Series, y_score: np.ndarray, outpath: Path) -> None:
    prob_true, prob_pred = calibration_curve(y_true, y_score, n_bins=10, strategy="uniform")

    plt.figure(figsize=(6, 4))
    plt.plot(prob_pred, prob_true, marker="o", label="Model")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    plt.xlabel("Mean Predicted Probability")
    plt.ylabel("Fraction of Positives")
    plt.title("Calibration Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, outpath: Path) -> None:
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks([0, 1], ["No Churn", "Churn"])
    plt.yticks([0, 1], ["No Churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_feature_importance_from_pipeline(pipeline: Pipeline, outpath: Path, top_n: int = 20) -> None:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(model, "feature_importances_"):
        return

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return

    fi = (
        pd.DataFrame({"feature": feature_names, "importance": model.feature_importances_})
        .sort_values("importance", ascending=False)
        .head(top_n)
    )

    plt.figure(figsize=(8, 6))
    plt.barh(fi["feature"][::-1], fi["importance"][::-1])
    plt.xlabel("Importance")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def save_shap_summary_from_pipeline(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    outpath: Path,
    top_n: int = 20,
) -> None:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    if not hasattr(model, "feature_importances_"):
        return

    try:
        X_transformed = preprocessor.transform(X_sample)
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_transformed)

        if isinstance(shap_values, list):
            shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

        plt.figure()
        shap.summary_plot(
            shap_values,
            X_transformed,
            feature_names=feature_names,
            max_display=top_n,
            show=False,
        )
        plt.tight_layout()
        plt.savefig(outpath, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:  # pragma: no cover - visualization best-effort
        logger.warning("Skipping SHAP summary plot: %s", e)


def save_shap_importance_table_from_pipeline(
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    outpath: Path,
    top_n: int = 30,
) -> None:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["preprocessor"]

    try:
        X_transformed = preprocessor.transform(X_sample)
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        return

    if hasattr(model, "feature_importances_"):
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_transformed)

            if isinstance(shap_values, list):
                shap_values = shap_values[1] if len(shap_values) > 1 else shap_values[0]

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
        except Exception as e:  # pragma: no cover - visualization best-effort
            logger.warning("Skipping SHAP TreeExplainer; falling back to feature_importances_: %s", e)
            mean_abs_shap = np.abs(getattr(model, "feature_importances_", np.zeros(len(feature_names))))
    elif hasattr(model, "coef_"):
        coef = getattr(model, "coef_", None)
        coef = coef[0] if coef is not None and coef.ndim > 1 else coef
        mean_abs_shap = np.abs(coef) if coef is not None else np.zeros(len(feature_names))
    else:
        return

    shap_df = (
        pd.DataFrame(
            {
                "feature": feature_names,
                "mean_abs_shap": mean_abs_shap,
            }
        )
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
    )

    shap_df.to_csv(outpath, index=False)


def log_eval_artifacts(
    model_name: str,
    pipeline: Pipeline,
    X_sample: pd.DataFrame,
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float,
    artifact_dir: Path,
) -> Dict[str, float]:
    artifact_dir = Path(artifact_dir)
    ensure_dir(artifact_dir)
    y_pred = (y_score >= threshold).astype(int)

    roc_path = artifact_dir / f"{model_name}_roc_curve.png"
    pr_path = artifact_dir / f"{model_name}_pr_curve.png"
    cal_path = artifact_dir / f"{model_name}_calibration_curve.png"
    cm_path = artifact_dir / f"{model_name}_confusion_matrix.png"
    fi_path = artifact_dir / f"{model_name}_feature_importance.png"
    shap_path = artifact_dir / f"{model_name}_shap_summary.png"
    shap_table_path = artifact_dir / f"{model_name}_shap_importance.csv"
    report_path = artifact_dir / f"{model_name}_classification_report.json"

    save_roc_curve(y_true, y_score, roc_path)
    save_pr_curve(y_true, y_score, pr_path)
    save_calibration_curve(y_true, y_score, cal_path)
    save_confusion_matrix(y_true, y_pred, cm_path)
    save_feature_importance_from_pipeline(pipeline, fi_path)
    save_shap_summary_from_pipeline(pipeline, X_sample, shap_path)
    save_shap_importance_table_from_pipeline(pipeline, X_sample, shap_table_path)

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    mlflow.log_artifact(str(roc_path))
    mlflow.log_artifact(str(pr_path))
    mlflow.log_artifact(str(cal_path))
    mlflow.log_artifact(str(cm_path))
    if fi_path.exists():
        mlflow.log_artifact(str(fi_path))
    if shap_path.exists():
        mlflow.log_artifact(str(shap_path))
    if shap_table_path.exists():
        mlflow.log_artifact(str(shap_table_path))
    mlflow.log_artifact(str(report_path))

    return report
