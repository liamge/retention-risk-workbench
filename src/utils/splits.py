from __future__ import annotations

import warnings
from typing import Optional

import pandas as pd
from sklearn.model_selection import train_test_split


def time_based_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    date_col: str,
    train_end: str,
    dev_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Time-based 3-way split:
      train: date <= train_end
      dev:   train_end < date <= dev_end
      test:  date > dev_end
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    if date_col not in X.columns:
        raise ValueError(f"{date_col} column required for time-based split.")

    tx = pd.to_datetime(X[date_col], errors="coerce")
    train_end_dt = pd.to_datetime(train_end)
    dev_end_dt = pd.to_datetime(dev_end)

    train_mask = tx <= train_end_dt
    dev_mask = (tx > train_end_dt) & (tx <= dev_end_dt)
    test_mask = tx > dev_end_dt

    if not train_mask.any() or not dev_mask.any() or not test_mask.any():
        raise ValueError("Time split produced empty partitions; adjust train_end/dev_end in config.")

    X_train = X.loc[train_mask].copy()
    y_train = y.loc[train_mask].copy()
    X_dev = X.loc[dev_mask].copy()
    y_dev = y.loc[dev_mask].copy()
    X_test = X.loc[test_mask].copy()
    y_test = y.loc[test_mask].copy()

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def _resolve_split_sizes(
    *,
    train_size: float | None = None,
    dev_size: float | None = None,
    test_size: float | None = None,
) -> tuple[float, float]:
    """Normalize split sizes so they sum to ~1.0 and stay positive."""

    default_train = 0.70
    default_dev = 0.15

    train_size = default_train if train_size is None else float(train_size)
    dev_size = default_dev if dev_size is None else float(dev_size)

    if test_size is None:
        test_size = 1.0 - train_size - dev_size
    else:
        test_size = float(test_size)

    if train_size <= 0 or dev_size <= 0 or test_size <= 0:
        raise ValueError("train_size, dev_size, and test_size must be positive and sum to < 1.0")

    total = train_size + dev_size + test_size
    if total > 1.0 + 1e-8:
        raise ValueError("train_size + dev_size + test_size must be <= 1.0")

    return dev_size, test_size


def random_three_way_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    random_state: int = 42,
    dev_size: float = 0.2,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Random 3-way split for non-temporal datasets.

    dev_size and test_size are fractions of the full dataset.
    Example:
      dev_size=0.2, test_size=0.2 -> 60/20/20 split
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    if not 0 < dev_size < 1:
        raise ValueError("dev_size must be between 0 and 1")

    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1")

    if dev_size + test_size >= 1:
        raise ValueError("dev_size + test_size must be less than 1")

    stratify_y = y if y.nunique() > 1 else None

    X_train_dev, X_test, y_train_dev, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=stratify_y,
        random_state=random_state,
    )

    dev_fraction_of_train_dev = dev_size / (1.0 - test_size)
    stratify_train_dev = y_train_dev if y_train_dev.nunique() > 1 else None

    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_dev,
        y_train_dev,
        test_size=dev_fraction_of_train_dev,
        stratify=stratify_train_dev,
        random_state=random_state,
    )

    return X_train, X_dev, X_test, y_train, y_dev, y_test


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    dataset_name: str,
    split_strategy: str | None = None,
    fallback_strategy: Optional[str] = None,
    random_state: int = 42,
    train_size: float | None = None,
    dev_size: float | None = None,
    test_size: float | None = None,
    date_col: Optional[str] = None,
    train_end: Optional[str] = None,
    dev_end: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Unified entry point. Always returns:
      X_train, X_dev, X_test, y_train, y_dev, y_test
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    strategy = split_strategy or ("time" if dataset_name.lower() == "kkbox" else "random")
    dev_size_resolved, test_size_resolved = _resolve_split_sizes(
        train_size=train_size,
        dev_size=dev_size,
        test_size=test_size,
    )

    if strategy == "time":
        if not date_col:
            raise ValueError("date_col is required for time-based splits")
        if not train_end or not dev_end:
            raise ValueError("train_end and dev_end are required for time-based splits")

        try:
            return time_based_split(
                X,
                y,
                date_col=date_col,
                train_end=train_end,
                dev_end=dev_end,
            )
        except ValueError:
            if fallback_strategy == "random":
                warnings.warn(
                    "Time-based split failed; falling back to random split using provided sizes.",
                    UserWarning,
                )
                return random_three_way_split(
                    X,
                    y,
                    random_state=random_state,
                    dev_size=dev_size_resolved,
                    test_size=test_size_resolved,
                )
            raise

    return random_three_way_split(
        X,
        y,
        random_state=random_state,
        dev_size=dev_size_resolved,
        test_size=test_size_resolved,
    )


def get_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    dataset_name: str,
    split_strategy: str | None = None,
    random_state: int = 42,
    train_size: float | None = None,
    dev_size: float | None = None,
    test_size: float | None = None,
    date_col: str = "transaction_date",
    dev_end: str | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Return only the held-out test split used for evaluation.

    For kkbox:
      test = rows where date_col > dev_end

    For non-temporal datasets:
      test is reconstructed with train_test_split using the same
      random_state and test_size used during training.
    """
    if len(X) != len(y):
        raise ValueError("X and y must have the same length")

    strategy = split_strategy or ("time" if dataset_name.lower() == "kkbox" else "random")
    dev_size_resolved, test_size_resolved = _resolve_split_sizes(
        train_size=train_size,
        dev_size=dev_size,
        test_size=test_size,
    )

    if strategy == "time":
        if date_col not in X.columns:
            raise ValueError(f"{date_col} column required for time-based split.")
        if not dev_end:
            raise ValueError("dev_end is required for kkbox test split.")

        tx = pd.to_datetime(X[date_col], errors="coerce")
        dev_end_dt = pd.to_datetime(dev_end)
        test_mask = tx > dev_end_dt

        if not test_mask.any():
            raise ValueError("Time split produced empty test partition; adjust dev_end in config.")

        return X.loc[test_mask].copy(), y.loc[test_mask].copy()

    stratify_y = y if y.nunique() > 1 else None
    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=test_size_resolved,
        stratify=stratify_y,
        random_state=random_state,
    )
    return X_test.copy(), y_test.copy()
