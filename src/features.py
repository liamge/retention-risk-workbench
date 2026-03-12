from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET_COL = "Churn"
ID_COL = "customerID"
TARGET_FLAG_COL = "ChurnFlag"


@dataclass
class FeatureArtifacts:
    X: pd.DataFrame
    y: pd.Series
    numeric_cols: List[str]
    categorical_cols: List[str]


def _safe_copy(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy(deep=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply first-pass churn feature engineering to the Telco dataset."""
    out = _safe_copy(df)

    if "TotalCharges" in out.columns:
        out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
        if "MonthlyCharges" in out.columns:
            out["TotalCharges"] = out["TotalCharges"].fillna(out["MonthlyCharges"])

    if TARGET_COL in out.columns:
        out[TARGET_FLAG_COL] = (out[TARGET_COL].astype(str).str.lower() == "yes").astype(int)

    # Lifetime / pricing features
    if {"TotalCharges", "tenure", "MonthlyCharges"}.issubset(out.columns):
        out["avg_monthly_spend"] = np.where(
            out["tenure"] > 0,
            out["TotalCharges"] / out["tenure"],
            out["MonthlyCharges"],
        )
        out["price_ratio"] = np.where(
            out["avg_monthly_spend"] > 0,
            out["MonthlyCharges"] / out["avg_monthly_spend"],
            1.0,
        )
        out["revenue_risk_proxy"] = out["MonthlyCharges"] * np.maximum(out["tenure"], 1)

    if "tenure" in out.columns:
        out["tenure_group"] = pd.cut(
            out["tenure"],
            bins=[-1, 0, 12, 24, 48, 72],
            labels=["new", "0-1yr", "1-2yr", "2-4yr", "4-6yr"],
        )
        out["customer_stage"] = pd.cut(
            out["tenure"],
            bins=[-1, 6, 24, 60, np.inf],
            labels=["new", "established", "loyal", "long_term"],
        )

    # Billing / contract features
    if "Contract" in out.columns:
        out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    if "PaymentMethod" in out.columns:
        out["auto_pay"] = (
            out["PaymentMethod"].astype("string").str.contains("automatic", case=False, na=False)
        ).astype(int)
        out["electronic_check"] = (out["PaymentMethod"] == "Electronic check").astype(int)
    if "PaperlessBilling" in out.columns:
        out["paperless"] = (out["PaperlessBilling"] == "Yes").astype(int)

    service_cols = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]
    present_service_cols = [c for c in service_cols if c in out.columns]
    if present_service_cols:
        out["num_services"] = (out[present_service_cols] == "Yes").sum(axis=1)
        out["service_adoption_rate"] = out["num_services"] / len(present_service_cols)
        out["multi_service_customer"] = (out["num_services"] >= 4).astype(int)

    if {"OnlineSecurity", "TechSupport"}.issubset(out.columns):
        out["has_support"] = (
            ((out["OnlineSecurity"] == "Yes") | (out["TechSupport"] == "Yes"))
        ).astype(int)

    if {"StreamingTV", "StreamingMovies"}.issubset(out.columns):
        out["has_streaming"] = (
            ((out["StreamingTV"] == "Yes") | (out["StreamingMovies"] == "Yes"))
        ).astype(int)

    if "InternetService" in out.columns:
        out["fiber_customer"] = (out["InternetService"] == "Fiber optic").astype(int)
    if "SeniorCitizen" in out.columns:
        out["senior_citizen"] = out["SeniorCitizen"].astype(int)

    if "MonthlyCharges" in out.columns:
        q75 = out["MonthlyCharges"].quantile(0.75)
        median = out["MonthlyCharges"].median()
        out["high_bill"] = (out["MonthlyCharges"] > q75).astype(int)
        if "tenure" in out.columns:
            out["low_tenure_high_bill"] = (
                (out["tenure"] <= 12) & (out["MonthlyCharges"] > median)
            ).astype(int)

    # Interaction features
    if {"InternetService", "Contract"}.issubset(out.columns):
        out["fiber_monthly"] = (
            (out["InternetService"] == "Fiber optic") & (out["Contract"] == "Month-to-month")
        ).astype(int)

    if {"Contract", "OnlineSecurity", "TechSupport"}.issubset(out.columns):
        support = ((out["OnlineSecurity"] == "Yes") | (out["TechSupport"] == "Yes")).astype(int)
        out["month_to_month_no_support"] = (
            (out["Contract"] == "Month-to-month") & (support == 0)
        ).astype(int)

    if {"MonthlyCharges", "tenure"}.issubset(out.columns):
        q75 = out["MonthlyCharges"].quantile(0.75)
        out["high_value_long_tenure"] = (
            (out["MonthlyCharges"] > q75) & (out["tenure"] > 24)
        ).astype(int)

    return out


def build_feature_frame(df: pd.DataFrame) -> FeatureArtifacts:
    engineered = engineer_features(df)

    if TARGET_FLAG_COL not in engineered.columns:
        raise ValueError(f"Expected target column '{TARGET_COL}' to create '{TARGET_FLAG_COL}'.")

    y = engineered[TARGET_FLAG_COL].copy()
    drop_cols = [c for c in [ID_COL, TARGET_COL, TARGET_FLAG_COL] if c in engineered.columns]
    X = engineered.drop(columns=drop_cols).copy()

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            numeric_cols.append(col)
        else:
            X[col] = X[col].astype("string").fillna("__missing__")
            categorical_cols.append(col)

    return FeatureArtifacts(X=X, y=y, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


def make_preprocessor(numeric_cols: List[str], categorical_cols: List[str]) -> ColumnTransformer:
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
