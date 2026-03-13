from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "is_churn"
ID_COL = "msno"


@dataclass
class KKBOXFeatureArtifacts:
    X: pd.DataFrame
    y: pd.Series
    numeric_cols: List[str]
    categorical_cols: List[str]


def engineer_kkbox_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(deep=True)

    # Date handling
    for col in ["transaction_date", "membership_expire_date", "registration_init_time"]:
        if col in out.columns:
            out[col] = pd.to_datetime(out[col], errors="coerce")

    if {"transaction_date", "membership_expire_date"}.issubset(out.columns):
        out["days_to_expire"] = (out["membership_expire_date"] - out["transaction_date"]).dt.days

    if {"registration_init_time", "transaction_date"}.issubset(out.columns):
        out["tenure_days"] = (out["transaction_date"] - out["registration_init_time"]).dt.days

    usage_cols = [
        "num_25",
        "num_50",
        "num_75",
        "num_985",
        "num_100",
        "num_unq",
        "total_secs",
    ]
    present_usage = [c for c in usage_cols if c in out.columns]
    if present_usage:
        out["total_plays"] = out[present_usage].sum(axis=1)
        if "total_secs" in out.columns:
            out["avg_seconds_per_play"] = np.where(out["total_plays"] > 0, out["total_secs"] / out["total_plays"], 0)

    if {"plan_list_price", "actual_amount_paid"}.issubset(out.columns):
        out["discount_rate"] = np.where(out["plan_list_price"] > 0, (out["plan_list_price"] - out["actual_amount_paid"]) / out["plan_list_price"], 0)

    if "is_auto_renew" in out.columns:
        out["auto_renew_int"] = out["is_auto_renew"].astype(int)
    if "is_cancel" in out.columns:
        out["cancel_int"] = out["is_cancel"].astype(int)

    return out


def build_kkbox_feature_frame(df: pd.DataFrame) -> KKBOXFeatureArtifacts:
    engineered = engineer_kkbox_features(df)

    if TARGET_COL not in engineered.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}'.")

    y = engineered[TARGET_COL].astype(int).copy()
    drop_cols = [c for c in [ID_COL, TARGET_COL] if c in engineered.columns]
    X = engineered.drop(columns=drop_cols).copy()

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            numeric_cols.append(col)
        else:
            X[col] = X[col].astype("string").fillna("__missing__")
            categorical_cols.append(col)

    return KKBOXFeatureArtifacts(X=X, y=y, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def make_kkbox_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    try:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        encoder = OneHotEncoder(handle_unknown="ignore", sparse=False)

    categorical_pipeline = Pipeline([
        ("encoder", encoder),
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_cols),
        ("cat", categorical_pipeline, categorical_cols),
    ])
