from __future__ import annotations

import pandas as pd

CAPPED_COLUMNS = [
    "total_secs_sum",
    "total_secs_avg",
    "songs_played_sum",
    "songs_played_avg",
    "num_unq_sum",
    "avg_amount_paid",
    "total_amount_paid",
    "avg_payment_plan_days",
    "days_since_last_log",
    "days_since_last_transaction",
    "days_until_membership_expire",
    "account_age_days",
    "latest_to_avg_total_secs_ratio",
    "latest_to_avg_songs_ratio",
    "latest_vs_avg_total_secs_delta",
    "latest_vs_avg_songs_delta",
]

RAW_DATE_COLUMNS = [
    "registration_init_date",
    "first_transaction_date",
    "last_transaction_date",
    "max_membership_expire_date",
    "latest_transaction_date",
    "latest_membership_expire_date",
    "first_log_date",
    "last_log_date",
    "latest_log_date",
]

CAPPED_PREFERRED_PAIRS = {
    "total_secs_sum": "total_secs_sum_capped",
    "total_secs_avg": "total_secs_avg_capped",
    "songs_played_sum": "songs_played_sum_capped",
    "songs_played_avg": "songs_played_avg_capped",
    "num_unq_sum": "num_unq_sum_capped",
    "avg_amount_paid": "avg_amount_paid_capped",
    "total_amount_paid": "total_amount_paid_capped",
    "avg_payment_plan_days": "avg_payment_plan_days_capped",
    "days_since_last_log": "days_since_last_log_capped",
    "days_since_last_transaction": "days_since_last_transaction_capped",
    "days_until_membership_expire": "days_until_membership_expire_capped",
    "account_age_days": "account_age_days_capped",
    "latest_to_avg_total_secs_ratio": "latest_to_avg_total_secs_ratio_capped",
    "latest_to_avg_songs_ratio": "latest_to_avg_songs_ratio_capped",
    "latest_vs_avg_total_secs_delta": "latest_vs_avg_total_secs_delta_capped",
    "latest_vs_avg_songs_delta": "latest_vs_avg_songs_delta_capped",
}


def add_capped_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with 1st/99th percentile capped heavy-tailed columns."""
    result = df.copy()
    for col in CAPPED_COLUMNS:
        if col in result.columns:
            series = pd.to_numeric(result[col], errors="coerce")
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            result[f"{col}_capped"] = series.clip(lower, upper)
    return result
