from __future__ import annotations

from typing import Dict

from src.features import build_feature_frame, make_preprocessor
from src.kkbox_features import build_kkbox_feature_frame, make_kkbox_preprocessor
from src.utils.io import read_table
from src.utils.splits import split_data


def build_dataset_objects(cfg: Dict):
    """
    Load data, apply feature engineering, and produce train/dev/test splits plus preprocessors.
    Returns (dataset_type, data_source, id_col, feature_artifacts, preprocessor, X_train, X_dev, X_test, y_train, y_dev, y_test).

    The returned preprocessor is an unfitted template configured with the feature columns; clone it
    when you need fresh instances for multiple pipelines (e.g., across Optuna trials).
    """
    data_cfg = cfg["data"]
    dataset_type = data_cfg.get("dataset_type", "telco").lower()
    id_col = data_cfg.get("id_col") or ("msno" if dataset_type == "kkbox" else "customerID")
    sample_frac = data_cfg.get("sample_frac")
    max_rows = data_cfg.get("max_rows")
    rng = data_cfg.get("sample_random_state", cfg["split"].get("random_state", 42))
    split_cfg = cfg["split"]
    train_size = split_cfg.get("train_size")
    dev_size = split_cfg.get("dev_size")
    test_size = split_cfg.get("test_size")
    random_state = split_cfg.get("random_state", 42)
    split_strategy = split_cfg.get("strategy", "time" if dataset_type == "kkbox" else "random")
    fallback_strategy = split_cfg.get("fallback")
    date_col_default = "latest_transaction_date" if dataset_type == "kkbox" else "transaction_date"
    date_col = split_cfg.get("date_col", date_col_default)

    if dataset_type == "kkbox":
        feature_path = data_cfg.get("feature_path")
        if not feature_path:
            raise ValueError("For dataset_type='kkbox', config must include data.feature_path")

        raw_df = read_table(feature_path)
        if id_col in raw_df.columns:
            raw_df = raw_df.set_index(id_col, drop=False)
        if sample_frac:
            raw_df = raw_df.sample(frac=float(sample_frac), random_state=rng)
        if max_rows:
            raw_df = raw_df.head(int(max_rows))

        target_col = data_cfg.get("target_col", "is_churn")
        if target_col not in raw_df.columns:
            raise ValueError(f"Target column '{target_col}' not found in KKBox feature table")

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
        if id_col in raw_df.columns:
            raw_df = raw_df.set_index(id_col, drop=False)
        if sample_frac:
            raw_df = raw_df.sample(frac=float(sample_frac), random_state=rng)
        if max_rows:
            raw_df = raw_df.head(int(max_rows))
        feature_artifacts = build_feature_frame(raw_df)
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
        preprocessor = make_preprocessor(
            feature_artifacts.numeric_cols,
            feature_artifacts.categorical_cols,
        )
        data_source = str(raw_path)

    return (
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
    )
