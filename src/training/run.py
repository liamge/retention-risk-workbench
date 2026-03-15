from __future__ import annotations

import argparse
import json
import copy
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline

from src.training.artifacts import (
    ensure_dir,
    save_candidate_results,
    save_metadata,
    save_model,
    save_test_artifacts,
)
from src.training.data import build_dataset_objects
from src.training.eval import evaluate_scores, log_eval_artifacts
from src.training.models import get_candidate_models
from src.utils.io import load_config
from src.utils.metrics import choose_threshold

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train churn models and save the champion pipeline.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    return parser.parse_args()


def apply_tuned_params(cfg: Dict, artifact_dir: Path) -> None:
    """
    If a tuning run produced best params, merge them into the model config
    so training uses the tuned hyperparameters.
    """
    tuning_cfg = cfg.get("tuning", {})
    algorithm = tuning_cfg.get("algorithm")
    if not algorithm:
        return

    best_path = Path(artifact_dir) / "tuning" / f"best_{algorithm}_params.json"
    if not best_path.exists():
        logger.info("No tuned params found at %s; using config defaults.", best_path)
        return

    try:
        with open(best_path, "r", encoding="utf-8") as f:
            best = json.load(f)
        tuned_params = best.get("best_params") or {}
        model_cfg = cfg.setdefault("models", {}).setdefault(algorithm, {})
        model_cfg.update(tuned_params)
        logger.info("Loaded tuned %s params from %s", algorithm, best_path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load tuned params from %s: %s", best_path, exc)


def train_champion(cfg: Dict) -> Tuple[str, Dict[str, float]]:
    cfg = copy.deepcopy(cfg)  # avoid mutating caller
    tracking_uri = cfg.get("mlflow", {}).get("tracking_uri", "file:./mlruns")
    experiment_name = cfg.get("mlflow", {}).get("experiment_name", "customer-churn-prod")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)

    artifact_dir = ensure_dir(cfg["artifacts"]["dir"])
    figure_dir = ensure_dir(artifact_dir / "figures")
    splits_dir = ensure_dir(artifact_dir / "splits")

    apply_tuned_params(cfg, artifact_dir)

    (
        dataset_type,
        data_source,
        id_col,
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
    candidate_models = get_candidate_models(cfg, class_ratio)

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
                threshold = choose_threshold(y_dev, dev_scores)
                dev_pred = (dev_scores >= threshold).astype(int)
                dev_best_f1 = float(f1_score(y_dev, dev_pred, zero_division=0))
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

        target_col = cfg["data"].get("target_col", "Churn" if dataset_type == "telco" else "is_churn")
        target_name = feature_artifacts.y.name or target_col
        save_test_artifacts(
            X_test=X_test,
            y_test=y_test,
            id_col=id_col,
            target_name=target_name,
            splits_dir=splits_dir,
        )

        results_df_path = artifact_dir / "candidate_model_results.csv"
        save_candidate_results(results_df, results_df_path)

        metrics_path = artifact_dir / "metrics.json"
        save_metadata({"champion": champion_name, "dev": results, "test": test_metrics}, metrics_path)

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

        model_path = artifact_dir / "model.pkl"
        save_model(champion_pipeline, model_path)
        metadata_path = artifact_dir / "model_metadata.json"
        save_metadata(metadata, metadata_path)

        sample_input = X_test.head(20)
        sample_output = champion_pipeline.predict_proba(sample_input)[:, 1]
        signature = infer_signature(sample_input, sample_output)

        mlflow.sklearn.log_model(champion_pipeline, artifact_path="model", signature=signature)
        mlflow.log_artifact(str(results_df_path))
        mlflow.log_artifact(str(metrics_path))
        mlflow.log_artifact(str(metadata_path))
        mlflow.set_tag("champion_model", champion_name)
        mlflow.set_tag("dataset_type", dataset_type)

        logger.info("Champion: %s", champion_name)
        logger.info("Test metrics: %s", json.dumps(test_metrics, indent=2))
        logger.info("Saved model bundle to %s", artifact_dir.resolve())

    return champion_name, test_metrics


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    cfg = load_config(args.config)
    train_champion(cfg)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Training run failed.")
        sys.exit(1)
