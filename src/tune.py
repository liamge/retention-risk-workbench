from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception as e:
    logger.warning("XGBoost unavailable: %s", e)
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except Exception as e:
    logger.warning("LightGBM unavailable: %s", e)
    LGBM_AVAILABLE = False
# Ensure repository root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features import build_feature_frame, make_preprocessor
from src.kkbox_features import build_kkbox_feature_frame, make_kkbox_preprocessor
from src.utils.io import ensure_dir, load_config, read_table
from src.utils.metrics import choose_threshold
from src.utils.splits import split_data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune XGBoost churn model with Optuna.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    if not (XGBOOST_AVAILABLE or LGBM_AVAILABLE):
        raise RuntimeError("Neither XGBoost nor LightGBM is available, so tuning cannot run.")

    args = args or parse_args()
    cfg = load_config(args.config)

    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    experiment_name = cfg.get("mlflow", {}).get("experiment_name", "customer-churn-prod")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    data_cfg = cfg["data"]
    data_path = data_cfg.get("raw_path") or data_cfg.get("feature_path")
    if not data_path:
        raise KeyError("Config is missing data.raw_path (or data.feature_path for KKBox).")

    dataset_type = cfg["data"].get("dataset_type", "telco").lower()
    split_cfg = cfg["split"]
    train_size = split_cfg.get("train_size")
    dev_size = split_cfg.get("dev_size")
    test_size = split_cfg.get("test_size")
    random_state = split_cfg.get("random_state", 42)
    split_strategy = split_cfg.get("strategy", "time" if dataset_type == "kkbox" else "random")
    fallback_strategy = split_cfg.get("fallback")
    date_col_default = "latest_transaction_date" if dataset_type == "kkbox" else "transaction_date"
    date_col = split_cfg.get("date_col", date_col_default)

    artifact_dir = ensure_dir(cfg["artifacts"]["dir"])
    tune_dir = ensure_dir(artifact_dir / "tuning")

    raw_df = read_table(data_path)
    target_col = cfg["data"].get("target_col", "Churn" if dataset_type == "telco" else "is_churn")

    if dataset_type == "kkbox":
        preprocessor_fn = make_kkbox_preprocessor
        if split_strategy == "time":
            X_raw = raw_df.drop(columns=[target_col])
            y_raw = raw_df[target_col].astype(int)
            X_train_raw, X_dev_raw, X_test_raw, y_train, y_dev, y_test = split_data(
                X_raw,
                y_raw,
                dataset_name=dataset_type,
                split_strategy=split_strategy,
                fallback_strategy=fallback_strategy,
                random_state=random_state,
                train_size=train_size,
                dev_size=dev_size,
                test_size=test_size,
                date_col=date_col,
                train_end=split_cfg.get("train_end"),
                dev_end=split_cfg.get("dev_end"),
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
            X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(
                X,
                y,
                dataset_name=dataset_type,
                split_strategy=split_strategy,
                fallback_strategy=fallback_strategy,
                random_state=random_state,
                train_size=train_size,
                dev_size=dev_size,
                test_size=test_size,
            )
    else:
        feature_artifacts = build_feature_frame(raw_df)
        preprocessor_fn = make_preprocessor
        X, y = feature_artifacts.X, feature_artifacts.y
        X_train, X_dev, X_test, y_train, y_dev, y_test = split_data(
            X,
            y,
            dataset_name=dataset_type,
            split_strategy=split_strategy,
            fallback_strategy=fallback_strategy,
            random_state=random_state,
            train_size=train_size,
            dev_size=dev_size,
            test_size=test_size,
        )
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

    run_name = f"tune-{algorithm}"
    best_params_filename = f"best_{algorithm}_params.json"

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("data_path", data_path)
        mlflow.log_param("n_trials", n_trials)
        mlflow.log_param("objective_metric", objective_metric)
        mlflow.log_param("algorithm", algorithm)

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

        best_params_path = tune_dir / best_params_filename
        with open(best_params_path, "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2)

        mlflow.log_metric("best_value", float(study.best_value))
        for key, value in study.best_params.items():
            mlflow.log_param(f"best_{key}", value)

        mlflow.log_artifact(str(tune_dir / "optuna_trials.csv"))
        mlflow.log_artifact(str(best_params_path))

        logger.info("Best trial summary:\n%s", json.dumps(best, indent=2))
        logger.info("Saved tuning outputs to %s", tune_dir.resolve())

    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Tuning run failed.")
        sys.exit(1)
