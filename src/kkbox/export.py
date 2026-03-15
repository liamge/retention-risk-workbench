from __future__ import annotations

import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
import yaml
from src.utils.io import ensure_dir, read_table

logger = logging.getLogger(__name__)


def derive_split_paths(base_output: Path, explicit_test: Path | None = None) -> Tuple[Path, Path]:
    """Return train/test parquet destinations alongside the base parquet."""
    test_out = explicit_test or base_output.with_name(base_output.stem + "_test.parquet")
    train_out = base_output.with_name(base_output.stem + "_train.parquet")
    return train_out, test_out


def split_for_artifacts(df: pd.DataFrame, split_cfg: dict, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a features dataframe into train/test using configured strategy."""
    strategy = split_cfg.get("strategy", "random")
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' present to perform split.")

    if strategy == "random":
        from sklearn.model_selection import train_test_split

        random_state = split_cfg.get("random_state", 42)
        train_size = split_cfg.get("train_size", 0.7)
        dev_size = split_cfg.get("dev_size", 0.15)
        test_size = 1.0 - train_size - dev_size
        if test_size <= 0:
            raise ValueError("train_size + dev_size must be less than 1.0 for random split.")

        stratify = df[target_col]
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            stratify=stratify,
            random_state=random_state,
        )
        return train_df, test_df

    if strategy == "time":
        dev_end = pd.to_datetime(split_cfg["dev_end"])
        date_col = split_cfg.get("date_col", "latest_transaction_date")
        if date_col not in df.columns:
            raise ValueError(f"Time-based split requires date column '{date_col}'.")

        date_series = pd.to_datetime(df[date_col], errors="coerce")
        train_dev_mask = date_series <= dev_end
        test_mask = date_series > dev_end

        if not train_dev_mask.any() or not test_mask.any():
            fallback = split_cfg.get("fallback", "random")
            if fallback == "random":
                import warnings

                warnings.warn(
                    "Time-based split produced empty partitions; falling back to stratified random split.",
                    UserWarning,
                )
                return split_for_artifacts(df, {"strategy": "random", **split_cfg}, target_col)
            raise ValueError(
                "Time-based split failed: train/dev or test is empty. Adjust dev_end or set fallback=random."
            )

        return df.loc[train_dev_mask].copy(), df.loc[test_mask].copy()

    raise ValueError(f"Unknown split strategy '{strategy}'. Expected 'random' or 'time'.")


def maybe_split_existing(
    output_path: Path,
    split_test: bool,
    reuse_existing: bool,
    test_output_path: Path | None,
    config_path: Path | None,
    target_col: str,
) -> bool:
    """
    If a cleaned dataset already exists and splits are missing, create them and short-circuit.
    Returns True when work was performed and caller should exit early.
    """
    if not split_test or not reuse_existing or not output_path.exists():
        return False

    if config_path is None:
        raise ValueError("split_test=True with existing dataset requires config_path to read split strategy.")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    train_out, test_out = derive_split_paths(output_path, test_output_path)
    if train_out.exists() and test_out.exists():
        logger.info("Cleaned dataset and splits already exist; skipping build.")
        return True

    logger.info("Found existing cleaned dataset at %s; creating splits using %s...", output_path, config_path)
    df = read_table(output_path)
    train_df, test_df = split_for_artifacts(df, cfg.get("split", {}), target_col)

    ensure_dir(train_out.parent)
    train_df.to_parquet(train_out, index=False)
    test_df.to_parquet(test_out, index=False)

    logger.info("Split counts -> train: %s, test: %s", len(train_df), len(test_df))
    logger.info("Train saved to: %s", train_out)
    logger.info("Test saved to:  %s", test_out)
    return True
