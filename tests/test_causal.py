import pandas as pd

from src.causal import (
    build_uplift_table,
    estimate_ate,
    estimate_cate_by_segment,
    recommend_policy,
    top_k_indices,
)


def make_toy_df():
    return pd.DataFrame(
        {
            "treatment": [1, 1, 1, 0, 0, 0, 1, 0],
            "outcome": [0, 0, 1, 1, 0, 1, 0, 1],  # 1 = churn, so lower is better
            "score": [0.9, 0.8, 0.7, 0.6, 0.55, 0.5, 0.45, 0.4],
            "segment": ["A", "A", "B", "B", "A", "B", "A", "B"],
        }
    )


def test_estimate_ate_reduces_churn():
    df = make_toy_df()
    ate = estimate_ate(df, treatment_col="treatment", outcome_col="outcome")
    assert round(ate.ate, 3) == -0.500  # treatment lowers churn rate
    assert ate.uplift_direction == "improves"


def test_estimate_cate_orders_by_effect_size():
    df = make_toy_df()
    cate = estimate_cate_by_segment(df, segment_cols="segment", treatment_col="treatment", outcome_col="outcome")
    assert not cate.empty
    # Largest absolute effect should be first
    assert cate.iloc[0]["ate"] == cate["ate"].abs().max()


def test_uplift_table_shapes_and_monotonicity():
    df = make_toy_df()
    uplift = build_uplift_table(df["score"], df["treatment"], df["outcome"], n_bins=4)
    assert len(uplift) == 4
    # bins sorted ascending by bin index
    assert list(uplift["bin"]) == sorted(uplift["bin"].tolist())


def test_policy_respects_budget_fraction():
    df = make_toy_df()
    uplift = build_uplift_table(df["score"], df["treatment"], df["outcome"], n_bins=4)
    policy = recommend_policy(uplift, treatment_cost=1.0, value_per_success=5.0, budget_fraction=0.5)
    treated_bins = policy[policy["treat"]]
    assert treated_bins["bin"].max() <= 1  # half the bins given budget_fraction=0.5


def test_top_k_indices_returns_expected_count():
    df = make_toy_df()
    idx = top_k_indices(df["score"], budget_fraction=0.25)
    assert len(idx) == max(1, int(len(df) * 0.25))
