from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import optuna
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, f1_score, precision_recall_curve, roc_auc_score
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
    parser = argparse.ArgumentParser(description="Tune XGBoost churn model with Optuna.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    return parser.parse_args()


def load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


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


def time_based_split(X, y, cfg: Dict):
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


def choose_threshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    if len(thresholds) == 0:
        return 0.5
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / np.clip(precision[:-1] + recall[:-1], 1e-9, None)
    best_idx = int(np.nanargmax(f1_scores))
    return float(thresholds[best_idx])


def main() -> None:
    if not (XGBOOST_AVAILABLE or LGBM_AVAILABLE):
        raise RuntimeError("Neither XGBoost nor LightGBM is available, so tuning cannot run.")

    args = parse_args()
    cfg = load_config(args.config)

    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    experiment_name = cfg.get("mlflow", {}).get("experiment_name", "customer-churn-prod")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    data_path = cfg["data"]["raw_path"]
    artifact_dir = ensure_dir(cfg["artifacts"]["dir"])
    tune_dir = ensure_dir(artifact_dir / "tuning")

    raw_df = pd.read_csv(data_path)
    dataset_type = cfg["data"].get("dataset_type", "telco")
    target_col = cfg["data"].get("target_col", "Churn" if dataset_type == "telco" else "is_churn")

    if dataset_type == "kkbox":
        feature_artifacts = build_kkbox_feature_frame(raw_df, target_col=target_col)
        splitter = time_based_split
        preprocessor_fn = make_kkbox_preprocessor
    else:
        feature_artifacts = build_feature_frame(raw_df)
        splitter = split_data
        preprocessor_fn = make_preprocessor

    X, y = feature_artifacts.X, feature_artifacts.y

    X_train, X_dev, X_test, y_train, y_dev, y_test = splitter(X, y, cfg)
    class_ratio = float((y_train == 0).sum() / max((y_train == 1).sum(), 1))

    tuning_cfg = cfg.get("tuning", {})
    n_trials = int(tuning_cfg.get("n_trials", 30))
    objective_metric = tuning_cfg.get("objective_metric", "roc_auc")
    algorithm = tuning_cfg.get("algorithm", "xgboost")
    if algorithm not in ("xgboost", "lightgbm"):
        raise ValueError("tuning.algorithm must be one of ['xgboost', 'lightgbm']")
    if algorithm == "xgboost" and not XGBOOST_AVAILABLE:
        raise RuntimeError("XGBoost not available but tuning.algorithm is xgboost")
    if algorithm == "lightgbm" and not LGBM_AVAILABLE:
        raise RuntimeError("LightGBM not available but tuning.algorithm is lightgbm")

    def objective(trial: optuna.Trial) -> float:
        preprocessor = preprocessor_fn(feature_artifacts.numeric_cols, feature_artifacts.categorical_cols)

        if algorithm == "xgboost":
            estimator = XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=cfg["split"]["random_state"],
                n_estimators=trial.suggest_int("n_estimators", 100, 800),
                max_depth=trial.suggest_int("max_depth", 3, 10),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                min_child_weight=trial.suggest_int("min_child_weight", 1, 12),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                gamma=trial.suggest_float("gamma", 0.0, 5.0),
                scale_pos_weight=trial.suggest_float("scale_pos_weight", max(0.5, class_ratio * 0.5), class_ratio * 2.0),
            )
        else:
            estimator = LGBMClassifier(
                objective="binary",
                n_estimators=trial.suggest_int("n_estimators", 100, 800),
                max_depth=trial.suggest_int("max_depth", -1, 12),
                num_leaves=trial.suggest_int("num_leaves", 16, 128),
                learning_rate=trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                subsample=trial.suggest_float("subsample", 0.6, 1.0),
                colsample_bytree=trial.suggest_float("colsample_bytree", 0.6, 1.0),
                reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                min_child_samples=trial.suggest_int("min_child_samples", 10, 100),
                class_weight={0: 1.0, 1: class_ratio},
                random_state=cfg["split"]["random_state"],
            )

        pipeline = Pipeline(
            [
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )
        pipeline.fit(X_train, y_train)

        dev_scores = pipeline.predict_proba(X_dev)[:, 1]
        threshold = choose_threshold(y_dev, dev_scores)
        dev_pred = (dev_scores >= threshold).astype(int)

        roc_auc = float(roc_auc_score(y_dev, dev_scores))
        pr_auc = float(average_precision_score(y_dev, dev_scores))
        f1 = float(f1_score(y_dev, dev_pred, zero_division=0))

        with mlflow.start_run(run_name=f"optuna_trial_{trial.number}", nested=True):
            mlflow.log_param("trial_number", trial.number)
            for key, value in estimator.get_params().items():
                if isinstance(value, (str, int, float, bool)) or value is None:
                    mlflow.log_param(key, value)
            mlflow.log_metric("roc_auc", roc_auc)
            mlflow.log_metric("pr_auc", pr_auc)
            mlflow.log_metric("f1", f1)
            mlflow.log_metric("threshold", threshold)

        trial.set_user_attr("threshold", threshold)
        trial.set_user_attr("roc_auc", roc_auc)
        trial.set_user_attr("pr_auc", pr_auc)
        trial.set_user_attr("f1", f1)

        if objective_metric == "f1":
            return f1
        if objective_metric == "pr_auc":
            return pr_auc
        return roc_auc

    with mlflow.start_run(run_name="tune-xgboost"):
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("objective_metric", objective_metric)

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=n_trials)

        best = {
            "best_value": study.best_value,
            "best_params": study.best_params,
            "best_trial_number": study.best_trial.number,
            "objective_metric": objective_metric,
            "best_threshold": study.best_trial.user_attrs.get("threshold", 0.5),
            "roc_auc": study.best_trial.user_attrs.get("roc_auc"),
            "pr_auc": study.best_trial.user_attrs.get("pr_auc"),
            "f1": study.best_trial.user_attrs.get("f1"),
        }

        trials_df = study.trials_dataframe()
        trials_df.to_csv(tune_dir / "optuna_trials.csv", index=False)

        with open(tune_dir / "best_xgboost_params.json", "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

        mlflow.log_metric("best_value", float(study.best_value))
        for key, value in study.best_params.items():
            mlflow.log_param(f"best_{key}", value)

        mlflow.log_artifact(str(tune_dir / "optuna_trials.csv"))
        mlflow.log_artifact(str(tune_dir / "best_xgboost_params.json"))

        print(json.dumps(best, indent=2))
        print(f"Saved tuning outputs to {tune_dir.resolve()}")


if __name__ == "__main__":
    main()
