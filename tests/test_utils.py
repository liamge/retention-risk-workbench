import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.utils.io import load_config, read_table
from src.utils.metrics import choose_threshold
from src.utils.splits import random_three_way_split, time_based_split
from src.utils.themes import map_feature_to_theme


def test_load_config_reads_yaml_and_json(tmp_path: Path):
    yaml_path = tmp_path / "config.yaml"
    yaml_path.write_text("a: 1\nb: test\n", encoding="utf-8")

    json_path = tmp_path / "config.json"
    json_path.write_text(json.dumps({"a": 2, "b": "json"}), encoding="utf-8")

    assert load_config(yaml_path) == {"a": 1, "b": "test"}
    assert load_config(json_path) == {"a": 2, "b": "json"}


def test_read_table_dispatches_by_suffix(monkeypatch, tmp_path: Path):
    csv_path = tmp_path / "data.csv"
    csv_path.write_text("x\n1\n", encoding="utf-8")
    pq_path = tmp_path / "data.pq"
    pq_path.write_bytes(b"")  # contents unused; we stub the reader
    txt_path = tmp_path / "data.txt"
    txt_path.write_text("x\n1\n", encoding="utf-8")

    csv_df = pd.DataFrame({"x": [1]})
    pq_df = pd.DataFrame({"y": [2]})

    monkeypatch.setattr("src.utils.io.pd.read_csv", lambda path: csv_df)
    monkeypatch.setattr("src.utils.io.pd.read_parquet", lambda path: pq_df)

    assert read_table(csv_path).equals(csv_df)
    assert read_table(pq_path).equals(pq_df)

    with pytest.raises(ValueError):
        read_table(txt_path)


def test_random_three_way_split_respects_proportions():
    X = pd.DataFrame({"a": range(10)})
    y = pd.Series([0, 1] * 5)

    X_train, X_dev, X_test, y_train, y_dev, y_test = random_three_way_split(
        X, y, random_state=123, dev_size=0.2, test_size=0.2
    )

    assert len(X_train) == 6
    assert len(X_dev) == 2
    assert len(X_test) == 2
    # stratification preserves class balance roughly
    assert y_train.sum() == 3


def test_time_based_split_orders_by_dates():
    X = pd.DataFrame(
        {
            "date_col": ["2024-01-01", "2024-02-01", "2024-03-01", "2024-04-01"],
            "feature": [1, 2, 3, 4],
        }
    )
    y = pd.Series([0, 0, 1, 1])

    X_train, X_dev, X_test, y_train, y_dev, y_test = time_based_split(
        X,
        y,
        date_col="date_col",
        train_end="2024-02-15",
        dev_end="2024-03-15",
    )

    assert list(X_train["feature"]) == [1, 2]
    assert list(X_dev["feature"]) == [3]
    assert list(X_test["feature"]) == [4]
    assert list(y_train) == [0, 0]
    assert list(y_dev) == [1]
    assert list(y_test) == [1]


def test_choose_threshold_returns_best_f1():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.4, 0.6, 0.9])

    threshold = choose_threshold(y_true, y_proba, metric="f1", step=0.05)

    assert 0.45 <= threshold <= 0.65

    with pytest.raises(ValueError):
        choose_threshold(y_true, y_proba, metric="roc_auc")


def test_map_feature_to_theme_matches_keywords():
    assert map_feature_to_theme("login_count") == "engagement"
    assert map_feature_to_theme("policy_status") == "production"
    assert map_feature_to_theme("months_since_signup") == "tenure"
    assert map_feature_to_theme("churn_probability") == "retention_risk"
    assert map_feature_to_theme("customer_state") == "profile"
    assert map_feature_to_theme("unknown_feature") == "other"
