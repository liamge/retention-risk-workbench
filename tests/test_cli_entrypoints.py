import argparse
import importlib
import json
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import pandas as pd


def test_train_cli_smoke(monkeypatch):
    called = {}

    def fake_load_config(path):
        called["config_path"] = path
        return {"ok": True}

    def fake_train(cfg):
        called["train_cfg"] = cfg
        return "model", {"roc_auc": 0.9}

    monkeypatch.setattr("src.training.run.load_config", fake_load_config)
    monkeypatch.setattr("src.training.run.train_champion", fake_train)

    train_module = importlib.import_module("src.cli.train")
    args = argparse.Namespace(config="dummy.yaml")

    assert train_module.main(args) == 0
    assert called["config_path"] == "dummy.yaml"
    assert called["train_cfg"] == {"ok": True}


def test_tune_cli_smoke(monkeypatch, tmp_path):
    cfg = {
        "data": {"raw_path": "data.csv", "dataset_type": "telco", "target_col": "Churn"},
        "split": {"dev_size": 0.2, "test_size": 0.2, "random_state": 7},
        "artifacts": {"dir": str(tmp_path / "artifacts")},
        "tuning": {"n_trials": 1, "objective_metric": "roc_auc", "algorithm": "xgboost"},
    }

    monkeypatch.setattr("src.cli.tune.load_config", lambda path: cfg)
    monkeypatch.setattr(
        "src.cli.tune.read_table",
        lambda path: pd.DataFrame({"Churn": [0, 1, 0], "f1": [0.1, 0.5, 0.9]}),
    )

    def fake_feature_frame(df, target_col="Churn"):
        return SimpleNamespace(
            X=df[[c for c in df.columns if c != target_col]],
            y=df[target_col],
            numeric_cols=[c for c in df.columns if c != target_col],
            categorical_cols=[],
        )

    monkeypatch.setattr("src.cli.tune.build_feature_frame", fake_feature_frame)
    monkeypatch.setattr("src.cli.tune.build_kkbox_feature_frame", fake_feature_frame)
    monkeypatch.setattr(
        "src.cli.tune.split_data",
        lambda X, y, **_: (X.iloc[:1], X.iloc[1:2], X.iloc[2:], y.iloc[:1], y.iloc[1:2], y.iloc[2:]),
    )

    class DummyStudy:
        def __init__(self):
            self.best_value = 1.0
            self.best_params = {"max_depth": 3}
            self.best_trial = SimpleNamespace(
                number=0,
                user_attrs={"threshold": 0.5, "roc_auc": 0.9, "pr_auc": 0.8, "f1": 0.7},
            )

        def optimize(self, *_args, **_kwargs):
            return None

        def trials_dataframe(self):
            return pd.DataFrame([{"value": self.best_value}])

    monkeypatch.setattr("src.cli.tune.optuna.create_study", lambda direction: DummyStudy())

    class DummyRun:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    for fn_name in (
        "set_tracking_uri",
        "set_experiment",
        "log_param",
        "log_metric",
        "log_artifact",
    ):
        monkeypatch.setattr(f"src.cli.tune.mlflow.{fn_name}", lambda *_, **__: None)

    monkeypatch.setattr("src.cli.tune.mlflow.start_run", lambda run_name=None, nested=False: DummyRun())
    monkeypatch.setattr("src.cli.tune.XGBOOST_AVAILABLE", True)
    monkeypatch.setattr("src.cli.tune.LGBM_AVAILABLE", True)

    tune_module = importlib.import_module("src.cli.tune")
    args = argparse.Namespace(config="dummy.yaml")

    assert tune_module.main(args) == 0


def test_predict_cli_smoke(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "threshold": 0.6,
        "champion_model": "dummy",
        "feature_columns": ["f1"],
        "dataset_type": "telco",
        "data_source": "data.csv",
    }
    (artifact_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    cfg = {
        "data": {"dataset_type": "telco", "target_col": "Churn"},
        "artifacts": {"dir": str(artifact_dir)},
    }

    def fake_ensure_dir(path: Path):
        target = tmp_path / Path(path).name
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr("src.cli.predict.ensure_dir", fake_ensure_dir)
    monkeypatch.setattr("src.cli.predict.load_config", lambda path: cfg)
    monkeypatch.setattr(
        "src.cli.predict.read_table",
        lambda path: pd.DataFrame({"customerID": ["a", "b"], "f1": [0.1, 0.9], "Churn": [0, 1]}),
    )

    class DummyModel:
        def predict_proba(self, X):
            return np.column_stack([np.zeros(len(X)), np.linspace(0.1, 0.9, len(X))])

    monkeypatch.setattr("src.cli.predict.joblib.load", lambda path: DummyModel())

    def fake_feature_frame(df, target_col="Churn"):
        return SimpleNamespace(X=df[[c for c in df.columns if c != target_col]], y=df[target_col])

    monkeypatch.setattr("src.cli.predict.build_feature_frame", fake_feature_frame)
    monkeypatch.setattr("src.cli.predict.build_kkbox_feature_frame", fake_feature_frame)
    monkeypatch.setattr("src.cli.predict.SHAP_AVAILABLE", False)

    predict_module = importlib.import_module("src.cli.predict")
    args = argparse.Namespace(config="dummy.yaml", input="dummy.csv")

    assert predict_module.main(args) == 0


def test_evaluate_cli_smoke(monkeypatch, tmp_path):
    artifact_dir = tmp_path / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    data_path = tmp_path / "data.csv"
    data_path.write_text("placeholder", encoding="utf-8")

    metadata = {
        "threshold": 0.5,
        "champion_model": "dummy",
        "feature_columns": ["f1"],
        "dataset_type": "telco",
        "data_source": str(data_path),
    }
    (artifact_dir / "model_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    cfg = {
        "data": {"raw_path": str(data_path), "dataset_type": "telco", "target_col": "Churn"},
        "split": {"dev_size": 0.2, "test_size": 0.2, "random_state": 7},
        "artifacts": {"dir": str(artifact_dir)},
    }

    def fake_ensure_dir(path: Path):
        target = tmp_path / Path(path).name
        target.mkdir(parents=True, exist_ok=True)
        return target

    monkeypatch.setattr("src.cli.evaluate.ensure_dir", fake_ensure_dir)
    monkeypatch.setattr("src.cli.evaluate.load_config", lambda path: cfg)
    monkeypatch.setattr(
        "src.cli.evaluate.read_table",
        lambda path: pd.DataFrame({"customerID": ["c1", "c2"], "f1": [0.2, 0.8], "Churn": [0, 1]}),
    )

    def fake_feature_frame(df):
        return SimpleNamespace(X=df[["f1"]], y=df["Churn"])

    monkeypatch.setattr("src.cli.evaluate.build_feature_frame", fake_feature_frame)
    monkeypatch.setattr("src.cli.evaluate.build_kkbox_feature_frame", fake_feature_frame)
    monkeypatch.setattr(
        "src.cli.evaluate.get_test_split", lambda X, y, **_: (X.copy(), y.copy())
    )

    class DummyModel:
        def predict_proba(self, X):
            return np.column_stack([1 - X["f1"].values, X["f1"].values])

    monkeypatch.setattr("src.cli.evaluate.joblib.load", lambda path: DummyModel())

    # Stub plotting functions to avoid heavy matplotlib usage in tests
    monkeypatch.setattr("src.cli.evaluate.RocCurveDisplay.from_predictions", lambda *_, **__: None)
    monkeypatch.setattr("src.cli.evaluate.PrecisionRecallDisplay.from_predictions", lambda *_, **__: None)
    monkeypatch.setattr("src.cli.evaluate.ConfusionMatrixDisplay.from_predictions", lambda *_, **__: None)

    class DummyFig:
        def tight_layout(self):
            return None

        def savefig(self, *_, **__):
            return None

    class DummyAx:
        def set_title(self, *_, **__):
            return None

    monkeypatch.setattr("src.cli.evaluate.plt.subplots", lambda figsize=None: (DummyFig(), DummyAx()))
    monkeypatch.setattr("src.cli.evaluate.plt.close", lambda fig=None: None)

    evaluate_module = importlib.import_module("src.cli.evaluate")
    args = argparse.Namespace(config="dummy.yaml", data_path=None, artifact_dir=None, report_dir=str(tmp_path / "reports"))

    assert evaluate_module.main(args) == 0
