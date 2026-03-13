from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import pandas as pd


@dataclass
class ExperimentSchema:
    treatment_col: str = "treatment"
    outcome_col: str = "outcome"
    propensity_col: Optional[str] = None
    segment_col: Optional[str] = None


def standardize_experiment_frame(df: pd.DataFrame, schema: ExperimentSchema) -> pd.DataFrame:
    """Validate and coerce the columns needed for causal estimation.

    - Treatment is coerced to int {0,1}
    - Outcome is coerced to float
    - Propensity, when present, is clipped away from 0/1 to avoid exploding weights.
    """

    missing = [c for c in [schema.treatment_col, schema.outcome_col] if c not in df]
    if missing:
        raise KeyError(f"Missing required columns: {', '.join(missing)}")

    out = df.copy()
    out[schema.treatment_col] = out[schema.treatment_col].astype(int)
    out[schema.outcome_col] = out[schema.outcome_col].astype(float)

    if schema.propensity_col and schema.propensity_col in out:
        out[schema.propensity_col] = out[schema.propensity_col].clip(1e-3, 1 - 1e-3)

    return out


def treatment_rate(df: pd.DataFrame, treatment_col: str = "treatment") -> float:
    """Share of records that received treatment."""

    if treatment_col not in df:
        raise KeyError(f"Missing treatment column '{treatment_col}'.")
    return float(df[treatment_col].mean())
