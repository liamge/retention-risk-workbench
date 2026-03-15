from __future__ import annotations

from typing import Iterable, Tuple

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def two_model_uplift(
    df: pd.DataFrame,
    feature_cols: Iterable[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    random_state: int = 42,
) -> Tuple[pd.Series, Pipeline, Pipeline]:
    """Train a two-model uplift estimator.

    A separate logistic regression is trained for the treated and control cohorts.
    The uplift score is the difference between the predicted positive outcome
    probability under treatment and control (treatment minus control).
    """

    if treatment_col not in df or outcome_col not in df:
        missing = {c for c in [treatment_col, outcome_col] if c not in df}
        raise KeyError(f"Missing required column(s): {', '.join(sorted(missing))}")

    feature_cols = list(feature_cols)
    X = df[feature_cols]
    y = df[outcome_col]

    treated_mask = df[treatment_col] == 1

    X_treated, X_control = X[treated_mask], X[~treated_mask]
    y_treated, y_control = y[treated_mask], y[~treated_mask]

    base_clf = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("model", LogisticRegression(max_iter=500)),
    ])

    treated_model = clone(base_clf)
    control_model = clone(base_clf)

    treated_model.fit(X_treated, y_treated)
    control_model.fit(X_control, y_control)

    uplift_scores = pd.Series(
        treated_model.predict_proba(X)[:, 1] - control_model.predict_proba(X)[:, 1],
        index=df.index,
        name="uplift_score",
    )

    return uplift_scores, treated_model, control_model


def build_uplift_table(
    scores: pd.Series,
    treatment: pd.Series,
    outcome: pd.Series,
    n_bins: int = 10,
) -> pd.DataFrame:
    """Construct an uplift table and Qini-style cumulative lift curve.

    Args:
        scores: Predicted uplift scores (higher means bigger benefit from treatment).
        treatment: Binary treatment assignments.
        outcome: Observed binary outcome (1 = event, e.g., churn).
        n_bins: Number of quantile bins for the curve.
    """

    df = pd.DataFrame({"score": scores, "treatment": treatment, "outcome": outcome})
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    df["bin"] = pd.qcut(df.index, q=n_bins, labels=False, duplicates="drop")

    rows = []
    cumulative = 0.0
    total_control_rate = df[df["treatment"] == 0]["outcome"].mean()

    for b, group in df.groupby("bin"):
        treated = group[group["treatment"] == 1]
        control = group[group["treatment"] == 0]

        treated_rate = treated["outcome"].mean() if not treated.empty else 0.0
        control_rate = control["outcome"].mean() if not control.empty else 0.0
        uplift = treated_rate - control_rate
        cumulative += uplift

        rows.append(
            {
                "bin": int(b),
                "n_records": int(len(group)),
                "treated_rate": float(treated_rate),
                "control_rate": float(control_rate),
                "uplift": float(uplift),
                "cumulative_uplift": float(cumulative),
                "control_baseline": float(total_control_rate),
            }
        )

    table = pd.DataFrame(rows).sort_values("bin").reset_index(drop=True)
    return table
