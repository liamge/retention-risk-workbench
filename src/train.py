from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import shap
import yaml
from mlflow.models import infer_signature
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except Exception as e:
    print(f"XGBoost unavailable: {e}")
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except Exception as e:
    print(f"LightGBM unavailable: {e}")
    LGBM_AVAILABLE = False

from src.features import build_feature_frame, make_preprocessor
from src.kkbox_features import build_kkbox_feature_frame, make_kkbox_preprocessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn models and save the champion pipeline.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported file type for {path}. Use CSV or Parquet.")


def split_data(X, y, cfg: Dict):
    random_state = cfg["split"]["random_state"]
    train_size = cfg["split"].get("train_size", 0.70)
    dev_size = cfg["split"].get("dev_size", 0.15)
    test_size = 1.0 - train_size - dev_size
    if test_size <= 0:
        raise ValueError("train_size + dev_size must be less than 1.0")

    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=(1.0 - train_size),
        stratify=y,
        random_state=random_state,
    )

    relative_test = test_size / (dev_size + test_size)
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=relative_test,
        stratify=y_temp,
        random_state=random_state,
    )
    return X_train, X_dev, X_test, y_train, y_dev, y_test


def time_based_split_df(df: pd.DataFrame, cfg: Dict, target_col: str):
    split_cfg = cfg["split"]
    train_end = pd.to_datetime(split_cfg["train_end"])
    dev_end = pd.to_datetime(split_cfg["dev_end"])
    date_col = split_cfg.get("date_col", "latest_transaction_date")

    if date_col not in df.columns:
        raise ValueError(f"{date_col} column required for time-based split.")

    tx = pd.to_datetime(df[date_col], errors="coerce")

    train_mask = tx <= train_end
    dev_mask = (tx > train_end) & (tx <= dev_end)
    test_mask = tx > dev_end

    if not train_mask.any() or not dev_mask.any() or not test_mask.any():
        if split_cfg.get("fallback", "random") == "random":
            warnings.warn(
                "Time-based split produced empty partitions; falling back to stratified random split.",
                UserWarning,
            )
            X = df.drop(columns=[target_col])
            y = df[target_col].astype(int)
            return split_data(X, y, {"split": {"random_state": split_cfg.get("random_state", 42)}})
        raise ValueError(
            f"Time split produced empty partitions using date_col={date_col}; "
            "adjust train_end/dev_end in config."
        )

    train_df = df.loc[train_mask].copy()
    dev_df = df.loc[dev_mask].copy()
    test_df = df.loc[test_mask].copy()

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].astype(int)
    X_dev = dev_df.drop(columns=[target_col])
    y_dev = dev_df[target_col].astype(int)
    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].astype(int)

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def choose_threshold(y_true: pd.Series, y_score: np.ndarray) -> Tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


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
    except Exception as e:
        print(f"Skipping SHAP summary plot: {e}")


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
        except Exception as e:
            print(f"Skipping SHAP TreeExplainer; falling back to feature_importances_: {e}")
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
    artifact_dir = ensure_dir(artifact_dir)
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


def build_dataset_objects(cfg: Dict):
    data_cfg = cfg["data"]
    dataset_type = data_cfg.get("dataset_type", "telco").lower()
    sample_frac = data_cfg.get("sample_frac")
    max_rows = data_cfg.get("max_rows")
    rng = data_cfg.get("sample_random_state", cfg["split"].get("random_state", 42))

    if dataset_type == "kkbox":
        feature_path = data_cfg.get("feature_path")
        if not feature_path:
            raise ValueError("For dataset_type='kkbox', config must include data.feature_path")

        raw_df = read_table(feature_path)
        if sample_frac:
            raw_df = raw_df.sample(frac=float(sample_frac), random_state=rng)
        if max_rows:
            raw_df = raw_df.head(int(max_rows))

        target_col = data_cfg.get("target_col", "is_churn")
        if target_col not in raw_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in KKBox feature table")

        split_strategy = cfg["split"].get("strategy", "time")
        if split_strategy == "time":
            X_train_raw, X_dev_raw, X_test_raw, y_train, y_dev, y_test = time_based_split_df(
                raw_df,
                cfg,
                target_col=target_col,
            )

            train_df = X_train_raw.copy()
            train_df[target_col] = y_train.values
            dev_df = X_dev_raw.copy()
            dev_df[target_col] = y_dev.values
            test_df = X_test_raw.copy()
            test_df[target_col] = y_test.values

            train_artifacts = build_kkbox_feature_frame(train_df, target_col=target_col)
            dev_artifacts = build_kkbox_feature_frame(dev_df, target_col=target_col)
            test_artifacts = build_kkbox_feature_frame(test_df, target_col=target_col)

            X_train, y_train = train_artifacts.X, train_artifacts.y
            X_dev, y_dev = dev_artifacts.X, dev_artifacts.y
            X_test, y_test = test_artifacts.X, test_artifacts.y

            feature_artifacts = train_artifacts
        else:
            feature_artifacts = build_kkbox_feature_frame(raw_df, target_col=target_col)
            X, y = feature_artifacts.X, feature_artifacts.y
            X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y, cfg)

        preprocessor = make_kkbox_preprocessor(
            feature_artifacts.numeric_cols,
            feature_artifacts.categorical_cols,
        )
        data_source = str(feature_path)

    else:
        raw_path = data_cfg.get("raw_path")
        if not raw_path:
            raise ValueError("For non-KKBox datasets, config must include data.raw_path")

        raw_df = read_table(raw_path)
        if sample_frac:
            raw_df = raw_df.sample(frac=float(sample_frac), random_state=rng)
        if max_rows:
            raw_df = raw_df.head(int(max_rows))
        feature_artifacts = build_feature_frame(raw_df)
        X, y = feature_artifacts.X, feature_artifacts.y
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(X, y, cfg)
        preprocessor = make_preprocessor(
            feature_artifacts.numeric_cols,
            feature_artifacts.categorical_cols,
        )
        data_source = str(raw_path)

    return dataset_type, data_source, feature_artifacts, preprocessor, X_train, X_dev, X_test, y_train, y_dev, y_test


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    experiment_name = cfg.get("mlflow", {}).get("experiment_name", "customer-churn-prod")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    artifact_dir = ensure_dir(cfg["artifacts"]["dir"])
    figure_dir = ensure_dir(artifact_dir / "figures")

    (
        dataset_type,
        data_source,
        feature_artifacts,
        preprocessor,
        X_train,
        X_dev,
        X_test,
        y_train,
        y_dev,
        y_test,
    ) = build_dataset_objects(cfg)

    class_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    candidate_models = {
        "logistic_regression": LogisticRegression(
            max_iter=cfg["models"]["logistic_regression"].get("max_iter", 2000)
        )
    }

    if XGBOOST_AVAILABLE:
        candidate_models["xgboost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=cfg["models"]["xgboost"].get("n_estimators", 400),
            learning_rate=cfg["models"]["xgboost"].get("learning_rate", 0.05),
            max_depth=cfg["models"]["xgboost"].get("max_depth", 4),
            subsample=cfg["models"]["xgboost"].get("subsample", 0.9),
            colsample_bytree=cfg["models"]["xgboost"].get("colsample_bytree", 0.9),
            min_child_weight=cfg["models"]["xgboost"].get("min_child_weight", 1),
            reg_lambda=cfg["models"]["xgboost"].get("reg_lambda", 1.0),
            scale_pos_weight=cfg["models"]["xgboost"].get("scale_pos_weight", class_ratio),
            random_state=cfg["split"].get("random_state", 42),
        )

    if LGBM_AVAILABLE:
        candidate_models["lightgbm"] = LGBMClassifier(
            objective="binary",
            n_estimators=cfg["models"]["lightgbm"].get("n_estimators", 400),
            learning_rate=cfg["models"]["lightgbm"].get("learning_rate", 0.05),
            max_depth=cfg["models"]["lightgbm"].get("max_depth", -1),
            num_leaves=cfg["models"]["lightgbm"].get("num_leaves", 31),
            subsample=cfg["models"]["lightgbm"].get("subsample", 0.9),
            colsample_bytree=cfg["models"]["lightgbm"].get("colsample_bytree", 0.9),
            reg_lambda=cfg["models"]["lightgbm"].get("reg_lambda", 1.0),
            min_child_samples=cfg["models"]["lightgbm"].get("min_child_samples", 20),
            class_weight={0: 1.0, 1: class_ratio},
            random_state=cfg["split"].get("random_state", 42),
            verbose=-1,
        )

    results = []
    pipelines = {}

    with mlflow.start_run(run_name=f"train-champion-{dataset_type}"):
        mlflow.log_params(
            {
                "dataset_type": dataset_type,
                "data_source": data_source,
                "split_strategy": cfg["split"].get("strategy"),
                "split_date_col": cfg["split"].get("date_col"),
                **({} if "train_size" not in cfg["split"] else {"train_size": cfg["split"]["train_size"]}),
                **({} if "dev_size" not in cfg["split"] else {"dev_size": cfg["split"]["dev_size"]}),
                **({} if "random_state" not in cfg["split"] else {"random_state": cfg["split"]["random_state"]}),
                **({} if "train_end" not in cfg["split"] else {"train_end": cfg["split"]["train_end"]}),
                **({} if "dev_end" not in cfg["split"] else {"dev_end": cfg["split"]["dev_end"]}),
            }
        )

        for model_name, estimator in candidate_models.items():
            with mlflow.start_run(run_name=model_name, nested=True):
                pipeline = Pipeline(
                    [
                        ("preprocessor", preprocessor),
                        ("model", estimator),
                    ]
                )
                pipeline.fit(X_train, y_train)
                dev_scores = pipeline.predict_proba(X_dev)[:, 1]
                threshold, dev_best_f1 = choose_threshold(y_dev, dev_scores)
                metrics = evaluate_scores(y_dev, dev_scores, threshold)
                metrics["dev_best_f1_scan"] = dev_best_f1
                metrics["model_name"] = model_name
                results.append(metrics)
                pipelines[model_name] = (pipeline, threshold)

                mlflow.log_param("model_name", model_name)
                if hasattr(estimator, "get_params"):
                    params = estimator.get_params()
                    for key, value in params.items():
                        if isinstance(value, (str, int, float, bool)) or value is None:
                            mlflow.log_param(key, value)

                for key, value in metrics.items():
                    if key != "model_name":
                        mlflow.log_metric(key, value)

                log_eval_artifacts(
                    model_name=model_name,
                    pipeline=pipeline,
                    X_sample=X_dev.head(300),
                    y_true=y_dev,
                    y_score=dev_scores,
                    threshold=threshold,
                    artifact_dir=figure_dir,
                )

        results_df = pd.DataFrame(results).sort_values("roc_auc", ascending=False)
        champion_name = results_df.iloc[0]["model_name"]
        champion_pipeline, champion_threshold = pipelines[champion_name]

        combined_X = pd.concat([X_train, X_dev], axis=0)
        combined_y = pd.concat([y_train, y_dev], axis=0)
        champion_pipeline.fit(combined_X, combined_y)

        test_scores = champion_pipeline.predict_proba(X_test)[:, 1]
        test_metrics = evaluate_scores(y_test, test_scores, champion_threshold)

        for key, value in test_metrics.items():
            mlflow.log_metric(f"test_{key}", value)

        log_eval_artifacts(
            model_name="champion_test",
            pipeline=champion_pipeline,
            X_sample=X_test.head(300),
            y_true=y_test,
            y_score=test_scores,
            threshold=champion_threshold,
            artifact_dir=figure_dir,
        )

        results_df.to_csv(artifact_dir / "candidate_model_results.csv", index=False)

        with open(artifact_dir / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"champion": champion_name, "dev": results, "test": test_metrics}, f, indent=2)

        metadata = {
            "champion_model": champion_name,
            "threshold": champion_threshold,
            "numeric_cols": feature_artifacts.numeric_cols,
            "categorical_cols": feature_artifacts.categorical_cols,
            "feature_columns": list(X_train.columns),
            "test_metrics": test_metrics,
            "dataset_type": dataset_type,
            "data_source": data_source,
            "time_split": (
                {
                    "train_end": cfg["split"]["train_end"],
                    "dev_end": cfg["split"]["dev_end"],
                    "date_col": cfg["split"].get("date_col"),
                }
                if "train_end" in cfg["split"]
                else None
            ),
        }

        joblib.dump(champion_pipeline, artifact_dir / "model.pkl")
        with open(artifact_dir / "model_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, default=str)

        sample_input = X_test.head(20)
        sample_output = champion_pipeline.predict_proba(sample_input)[:, 1]
        signature = infer_signature(sample_input, sample_output)

        mlflow.sklearn.log_model(champion_pipeline, artifact_path="model", signature=signature)
        mlflow.log_artifact(str(artifact_dir / "candidate_model_results.csv"))
        mlflow.log_artifact(str(artifact_dir / "metrics.json"))
        mlflow.log_artifact(str(artifact_dir / "model_metadata.json"))
        mlflow.set_tag("champion_model", champion_name)
        mlflow.set_tag("dataset_type", dataset_type)

        print("Champion:", champion_name)
        print("Test metrics:", json.dumps(test_metrics, indent=2))
        print(f"Saved model bundle to {artifact_dir.resolve()}")


if __name__ == "__main__":
    main()