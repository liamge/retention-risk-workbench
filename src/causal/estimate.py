from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np
import pandas as pd


@dataclass
class ATEstimate:
    ate: float
    treated_mean: float
    control_mean: float
    uplift_direction: str


def estimate_ate(
    df: pd.DataFrame,
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    weight_col: Optional[str] = None,
    outcome_higher_is_better: bool = False,
) -> ATEstimate:
    """Compute the average treatment effect using a simple difference in means.

    Args:
        df: Experimental or observational dataset with treatment and outcome columns.
        treatment_col: Name of the binary treatment indicator column (1 = treated).
        outcome_col: Name of the outcome column. Assumed to be numeric or binary.
        weight_col: Optional column containing observation weights (e.g., IPW).
        outcome_higher_is_better: If False (default), a lower outcome is better
            (common for churn where 1 = churn). This is only used to label the
            uplift_direction field for reporting.

    Returns:
        ATEstimate dataclass with treated/control means and ATE.
    """

    if treatment_col not in df or outcome_col not in df:
        missing = {c for c in [treatment_col, outcome_col] if c not in df}
        raise KeyError(f"Missing required column(s): {', '.join(sorted(missing))}")

    weights = df[weight_col] if weight_col and weight_col in df else None

    treated = df[df[treatment_col] == 1][outcome_col]
    control = df[df[treatment_col] == 0][outcome_col]

    treated_mean = float(np.average(treated, weights=weights.loc[treated.index] if weights is not None else None))
    control_mean = float(np.average(control, weights=weights.loc[control.index] if weights is not None else None))

    ate = treated_mean - control_mean
    direction = "improves" if (ate < 0 and not outcome_higher_is_better) or (ate > 0 and outcome_higher_is_better) else "worsens"

    return ATEstimate(
        ate=float(ate),
        treated_mean=float(treated_mean),
        control_mean=float(control_mean),
        uplift_direction=direction,
    )


def estimate_cate_by_segment(
    df: pd.DataFrame,
    segment_cols: Iterable[str],
    treatment_col: str = "treatment",
    outcome_col: str = "outcome",
    weight_col: Optional[str] = None,
    outcome_higher_is_better: bool = False,
) -> pd.DataFrame:
    """Compute conditional average treatment effects for each segment combination.

    Returns a dataframe sorted by absolute ATE so the largest effects rise to the top.
    """

    if isinstance(segment_cols, str):
        segment_cols = [segment_cols]

    missing = [c for c in segment_cols if c not in df]
    if missing:
        raise KeyError(f"Missing segment column(s): {', '.join(missing)}")

    records = []
    for seg_vals, group in df.groupby(list(segment_cols)):
        est = estimate_ate(
            group,
            treatment_col=treatment_col,
            outcome_col=outcome_col,
            weight_col=weight_col,
            outcome_higher_is_better=outcome_higher_is_better,
        )
        seg_dict: Dict[str, object] = {}
        if isinstance(seg_vals, tuple):
            for idx, col in enumerate(segment_cols):
                seg_dict[col] = seg_vals[idx]
        else:
            seg_dict[segment_cols[0]] = seg_vals

        records.append(
            {
                **seg_dict,
                "ate": est.ate,
                "treated_mean": est.treated_mean,
                "control_mean": est.control_mean,
                "uplift_direction": est.uplift_direction,
            }
        )

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.reindex(result["ate"].abs().sort_values(ascending=False).index)
    return result


def inverse_propensity_weights(df: pd.DataFrame, treatment_col: str = "treatment", propensity_col: str = "propensity") -> pd.Series:
    """Calculate inverse propensity weights for doubly robust estimators.

    Values are clipped to avoid extreme weights when propensities are near 0 or 1.
    """

    if propensity_col not in df:
        raise KeyError(f"Missing propensity column '{propensity_col}'.")

    prop = df[propensity_col].clip(1e-3, 1 - 1e-3)
    treat = df[treatment_col]
    return (treat / prop) + ((1 - treat) / (1 - prop))
