from __future__ import annotations

import numpy as np
import pandas as pd


def recommend_policy(
    uplift_table: pd.DataFrame,
    treatment_cost: float = 0.0,
    value_per_success: float = 1.0,
    budget_fraction: float = 1.0,
) -> pd.DataFrame:
    """Translate an uplift table into an actionable targeting policy.

    Args:
        uplift_table: Output from build_uplift_table with columns `bin`, `uplift`, and `n_records`.
        treatment_cost: Cost per treated customer (e.g., incentive or outreach cost).
        value_per_success: Business value of preventing churn (or achieving the desired outcome).
        budget_fraction: Portion of customers to target based on the highest uplift bins.
    """

    if uplift_table.empty:
        return uplift_table

    max_bin = int(np.floor((len(uplift_table) - 1) * budget_fraction))
    eligible_bins = uplift_table.sort_values("bin").head(max_bin + 1).copy()

    eligible_bins["expected_successes"] = eligible_bins["uplift"] * eligible_bins["n_records"]
    eligible_bins["expected_gain"] = (eligible_bins["expected_successes"] * value_per_success) - (
        eligible_bins["n_records"] * treatment_cost
    )
    eligible_bins["cumulative_gain"] = eligible_bins["expected_gain"].cumsum()
    eligible_bins["treat"] = True

    holdout_bins = uplift_table[~uplift_table["bin"].isin(eligible_bins["bin"])].copy()
    holdout_bins["expected_successes"] = 0.0
    holdout_bins["expected_gain"] = 0.0
    holdout_bins["cumulative_gain"] = float(eligible_bins["expected_gain"].sum())
    holdout_bins["treat"] = False

    policy = pd.concat([eligible_bins, holdout_bins], axis=0).sort_values("bin").reset_index(drop=True)
    return policy


def top_k_indices(scores: pd.Series, budget_fraction: float = 0.2) -> pd.Index:
    """Return indices for the highest uplift scores within a budget fraction."""

    k = max(1, int(len(scores) * budget_fraction))
    return scores.sort_values(ascending=False).head(k).index
