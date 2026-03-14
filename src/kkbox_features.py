from pathlib import Path
from typing import List

import duckdb
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import FeatureArtifacts

DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "train.csv"
MEMBERS_PATH = DATA_DIR / "members.csv"
TRANSACTIONS_PATH = DATA_DIR / "transactions.csv"
USER_LOGS_PATH = DATA_DIR / "user_logs.csv"

DUCKDB_PATH = OUTPUT_DIR / "kkbox_work.duckdb"
TEMP_DIR = OUTPUT_DIR / "duckdb_tmp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

BASE_OUTPUT_PATH = OUTPUT_DIR / "kkbox_features_base.parquet"
OUTPUT_PATH = OUTPUT_DIR / "kkbox_features_duckdb_cleaned.parquet"

DEFAULT_TARGET_COL = "is_churn"
DEFAULT_ID_COL = "msno"

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


def _sql_path(path: str | Path) -> str:
    return str(Path(path).resolve()).replace("\\", "/")


def _table_exists(con: duckdb.DuckDBPyConnection, table_name: str) -> bool:
    result = con.execute(
        f"""
        SELECT COUNT(*)
        FROM information_schema.tables
        WHERE table_schema = 'main'
          AND table_name = '{table_name}'
        """
    ).fetchone()
    return bool(result[0])


def _run_stage(
    con: duckdb.DuckDBPyConnection,
    table_name: str,
    sql: str,
    reuse_existing: bool = True,
) -> None:
    if reuse_existing and _table_exists(con, table_name):
        print(f"Reusing {table_name}...")
        return
    print(f"Building {table_name}...")
    con.execute(sql)


def build_feature_table(
    train_path: str | Path,
    members_path: str | Path,
    transactions_path: str | Path,
    user_logs_path: str | Path,
    output_path: str | Path,
    base_output_path: str | Path | None = None,
    add_capped_features: bool = True,
    reuse_existing: bool = True,
    force_recap_only: bool = False,
    split_test: bool = False,
    test_date_col: str = "latest_transaction_date",
    test_cutoff: str | None = None,
    test_output_path: str | Path | None = None,
) -> None:
    train_path = _sql_path(train_path)
    members_path = _sql_path(members_path)
    transactions_path = _sql_path(transactions_path)
    user_logs_path = _sql_path(user_logs_path)
    output_path = Path(output_path)
    base_output_path = Path(base_output_path) if base_output_path is not None else None

    if force_recap_only:
        if base_output_path is None or not base_output_path.exists():
            raise ValueError("force_recap_only=True requires an existing base_output_path parquet file.")
        print(f"Loading existing base feature parquet from {base_output_path}...")
        base_df = pd.read_parquet(base_output_path)
        if add_capped_features:
            for col in CAPPED_COLUMNS:
                if col in base_df.columns:
                    series = pd.to_numeric(base_df[col], errors="coerce")
                    lower = series.quantile(0.01)
                    upper = series.quantile(0.99)
                    base_df[f"{col}_capped"] = series.clip(lower, upper)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        base_df.to_parquet(output_path, index=False)
        print(f"Saved cleaned feature table to: {output_path}")
        return

    con = duckdb.connect(str(DUCKDB_PATH))
    con.execute("SET threads = 1;")
    con.execute("SET preserve_insertion_order = false;")
    con.execute(f"SET temp_directory = '{_sql_path(TEMP_DIR)}';")
    con.execute("SET max_temp_directory_size = '250GiB';")
    con.execute("SET memory_limit = '6GB';")
    con.execute("PRAGMA enable_progress_bar;")

    _run_stage(
        con,
        "train_base",
        f"""
        CREATE TABLE train_base AS
        SELECT
            CAST(msno AS VARCHAR) AS msno,
            CAST(is_churn AS INTEGER) AS is_churn
        FROM read_csv(
            '{train_path}',
            columns={{'msno': 'VARCHAR', 'is_churn': 'INTEGER'}},
            header=True
        );
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "members_base",
        f"""
        CREATE TABLE members_base AS
        SELECT
            CAST(msno AS VARCHAR) AS msno,

            CASE
                WHEN TRY_CAST(city AS INTEGER) BETWEEN 1 AND 25 THEN TRY_CAST(city AS INTEGER)
                ELSE NULL
            END AS city,

            CASE
                WHEN TRY_CAST(bd AS INTEGER) BETWEEN 10 AND 100 THEN TRY_CAST(bd AS INTEGER)
                ELSE NULL
            END AS age,

            CASE
                WHEN TRY_CAST(bd AS INTEGER) IS NULL THEN 1
                WHEN TRY_CAST(bd AS INTEGER) NOT BETWEEN 10 AND 100 THEN 1
                ELSE 0
            END AS age_missing_or_invalid,

            CASE
                WHEN lower(trim(CAST(gender AS VARCHAR))) = 'male' THEN 1
                ELSE 0
            END AS gender_male,

            CASE
                WHEN lower(trim(CAST(gender AS VARCHAR))) = 'female' THEN 1
                ELSE 0
            END AS gender_female,

            CASE
                WHEN gender IS NULL OR trim(CAST(gender AS VARCHAR)) = '' THEN 1
                ELSE 0
            END AS gender_missing,

            TRY_CAST(registered_via AS INTEGER) AS registered_via,
            STRPTIME(CAST(registration_init_time AS VARCHAR), '%Y%m%d') AS registration_init_date
        FROM read_csv(
            '{members_path}',
            columns={{
                'msno': 'VARCHAR',
                'city': 'VARCHAR',
                'bd': 'VARCHAR',
                'gender': 'VARCHAR',
                'registered_via': 'VARCHAR',
                'registration_init_time': 'VARCHAR'
            }},
            header=True
        );
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "transactions_clean",
        f"""
        CREATE TABLE transactions_clean AS
        SELECT
            CAST(msno AS VARCHAR) AS msno,

            CASE
                WHEN TRY_CAST(payment_method_id AS INTEGER) >= 0 THEN TRY_CAST(payment_method_id AS INTEGER)
                ELSE NULL
            END AS payment_method_id,

            CASE
                WHEN TRY_CAST(payment_plan_days AS DOUBLE) BETWEEN 0 AND 3650
                THEN TRY_CAST(payment_plan_days AS DOUBLE)
                ELSE NULL
            END AS payment_plan_days,

            CASE
                WHEN TRY_CAST(plan_list_price AS DOUBLE) BETWEEN 0 AND 100000
                THEN TRY_CAST(plan_list_price AS DOUBLE)
                ELSE NULL
            END AS plan_list_price,

            CASE
                WHEN TRY_CAST(actual_amount_paid AS DOUBLE) BETWEEN 0 AND 100000
                THEN TRY_CAST(actual_amount_paid AS DOUBLE)
                ELSE NULL
            END AS actual_amount_paid,

            CASE
                WHEN TRY_CAST(is_auto_renew AS INTEGER) IN (0, 1)
                THEN TRY_CAST(is_auto_renew AS INTEGER)
                ELSE 0
            END AS is_auto_renew,

            STRPTIME(CAST(transaction_date AS VARCHAR), '%Y%m%d') AS transaction_date,
            STRPTIME(CAST(membership_expire_date AS VARCHAR), '%Y%m%d') AS membership_expire_date,

            CASE
                WHEN TRY_CAST(is_cancel AS INTEGER) IN (0, 1)
                THEN TRY_CAST(is_cancel AS INTEGER)
                ELSE 0
            END AS is_cancel
        FROM read_csv(
            '{transactions_path}',
            columns={{
                'msno': 'VARCHAR',
                'payment_method_id': 'VARCHAR',
                'payment_plan_days': 'VARCHAR',
                'plan_list_price': 'VARCHAR',
                'actual_amount_paid': 'VARCHAR',
                'is_auto_renew': 'VARCHAR',
                'transaction_date': 'VARCHAR',
                'membership_expire_date': 'VARCHAR',
                'is_cancel': 'VARCHAR'
            }},
            header=True
        )
        WHERE STRPTIME(CAST(transaction_date AS VARCHAR), '%Y%m%d') IS NOT NULL;
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "transactions_ranked",
        """
        CREATE TABLE transactions_ranked AS
        SELECT
            *,
            ROW_NUMBER() OVER (
                PARTITION BY msno
                ORDER BY transaction_date DESC, membership_expire_date DESC
            ) AS txn_rn_desc,
            LAG(transaction_date) OVER (
                PARTITION BY msno ORDER BY transaction_date
            ) AS prev_transaction_date,
            LAG(membership_expire_date) OVER (
                PARTITION BY msno ORDER BY transaction_date
            ) AS prev_expire_date
        FROM transactions_clean;
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "transactions_features",
        """
        CREATE TABLE transactions_features AS
        SELECT
            msno,
            COUNT(*) AS txn_count,
            MIN(transaction_date) AS first_transaction_date,
            MAX(transaction_date) AS last_transaction_date,
            MAX(membership_expire_date) AS max_membership_expire_date,

            AVG(payment_plan_days) AS avg_payment_plan_days,
            MAX(payment_plan_days) AS max_payment_plan_days,

            SUM(plan_list_price) AS total_list_price,
            SUM(actual_amount_paid) AS total_amount_paid,
            AVG(actual_amount_paid) AS avg_amount_paid,
            MAX(actual_amount_paid) AS max_amount_paid,

            AVG(COALESCE(is_auto_renew, 0)) AS auto_renew_rate,
            AVG(COALESCE(is_cancel, 0)) AS cancel_rate,
            SUM(COALESCE(is_cancel, 0)) AS cancel_count,

            AVG(
                CASE
                    WHEN plan_list_price > 0
                    THEN actual_amount_paid / plan_list_price
                    ELSE NULL
                END
            ) AS avg_paid_to_list_ratio,

            AVG(
                CASE
                    WHEN prev_transaction_date IS NOT NULL
                    THEN DATE_DIFF('day', prev_transaction_date, transaction_date)
                    ELSE NULL
                END
            ) AS avg_days_between_transactions,

            AVG(
                CASE
                    WHEN prev_expire_date IS NOT NULL
                    THEN DATE_DIFF('day', prev_expire_date, transaction_date)
                    ELSE NULL
                END
            ) AS avg_gap_from_prev_expire_to_txn,

            SUM(
                CASE
                    WHEN prev_expire_date IS NOT NULL
                         AND transaction_date > prev_expire_date
                    THEN 1 ELSE 0
                END
            ) AS post_expiry_renewal_count,

            SUM(
                CASE
                    WHEN prev_expire_date IS NOT NULL
                         AND transaction_date <= prev_expire_date
                    THEN 1 ELSE 0
                END
            ) AS early_renewal_count
        FROM transactions_ranked
        GROUP BY msno;
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "latest_transaction_features",
        """
        CREATE TABLE latest_transaction_features AS
        SELECT
            msno,
            payment_method_id AS latest_payment_method_id,
            payment_plan_days AS latest_payment_plan_days,
            plan_list_price AS latest_plan_list_price,
            actual_amount_paid AS latest_actual_amount_paid,
            is_auto_renew AS latest_is_auto_renew,
            is_cancel AS latest_is_cancel,
            transaction_date AS latest_transaction_date,
            membership_expire_date AS latest_membership_expire_date
        FROM transactions_ranked
        WHERE txn_rn_desc = 1;
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "user_logs_clean",
        f"""
        CREATE TABLE user_logs_clean AS
        SELECT
            CAST(msno AS VARCHAR) AS msno,
            STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') AS log_date,

            CASE WHEN TRY_CAST(num_25 AS DOUBLE) BETWEEN 0 AND 100000 THEN TRY_CAST(num_25 AS DOUBLE) ELSE 0 END AS num_25,
            CASE WHEN TRY_CAST(num_50 AS DOUBLE) BETWEEN 0 AND 100000 THEN TRY_CAST(num_50 AS DOUBLE) ELSE 0 END AS num_50,
            CASE WHEN TRY_CAST(num_75 AS DOUBLE) BETWEEN 0 AND 100000 THEN TRY_CAST(num_75 AS DOUBLE) ELSE 0 END AS num_75,
            CASE WHEN TRY_CAST(num_985 AS DOUBLE) BETWEEN 0 AND 100000 THEN TRY_CAST(num_985 AS DOUBLE) ELSE 0 END AS num_985,
            CASE WHEN TRY_CAST(num_100 AS DOUBLE) BETWEEN 0 AND 100000 THEN TRY_CAST(num_100 AS DOUBLE) ELSE 0 END AS num_100,
            CASE WHEN TRY_CAST(num_unq AS DOUBLE) BETWEEN 0 AND 100000 THEN TRY_CAST(num_unq AS DOUBLE) ELSE 0 END AS num_unq,
            CASE WHEN TRY_CAST(total_secs AS DOUBLE) BETWEEN 0 AND 86400 THEN TRY_CAST(total_secs AS DOUBLE) ELSE 0 END AS total_secs
        FROM read_csv(
            '{user_logs_path}',
            columns={{
                'msno': 'VARCHAR',
                'date': 'VARCHAR',
                'num_25': 'VARCHAR',
                'num_50': 'VARCHAR',
                'num_75': 'VARCHAR',
                'num_985': 'VARCHAR',
                'num_100': 'VARCHAR',
                'num_unq': 'VARCHAR',
                'total_secs': 'VARCHAR'
            }},
            header=True
        )
        WHERE STRPTIME(CAST(date AS VARCHAR), '%Y%m%d') IS NOT NULL;
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "user_logs_scored",
        """
        CREATE TABLE user_logs_scored AS
        SELECT
            *,
            COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                + COALESCE(num_985, 0) + COALESCE(num_100, 0) AS songs_played,

            CASE
                WHEN (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                    + COALESCE(num_985, 0) + COALESCE(num_100, 0)) > 0
                THEN COALESCE(num_100, 0) * 1.0 /
                    (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                    + COALESCE(num_985, 0) + COALESCE(num_100, 0))
                ELSE 0
            END AS completion_rate,

            CASE
                WHEN (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                    + COALESCE(num_985, 0) + COALESCE(num_100, 0)) > 0
                THEN COALESCE(num_25, 0) * 1.0 /
                    (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                    + COALESCE(num_985, 0) + COALESCE(num_100, 0))
                ELSE 0
            END AS skip_rate,

            CASE
                WHEN (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                    + COALESCE(num_985, 0) + COALESCE(num_100, 0)) > 0
                THEN COALESCE(num_985, 0) * 1.0 /
                    (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                    + COALESCE(num_985, 0) + COALESCE(num_100, 0))
                ELSE 0
            END AS near_completion_rate,

            CASE
                WHEN (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                    + COALESCE(num_985, 0) + COALESCE(num_100, 0)) > 0
                THEN 1 - (
                    COALESCE(num_unq, 0) * 1.0 /
                    (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                    + COALESCE(num_985, 0) + COALESCE(num_100, 0))
                )
                ELSE 0
            END AS repeat_ratio,

            CASE
                WHEN (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                    + COALESCE(num_985, 0) + COALESCE(num_100, 0)) > 0
                THEN COALESCE(total_secs, 0) * 1.0 /
                    (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                    + COALESCE(num_985, 0) + COALESCE(num_100, 0))
                ELSE 0
            END AS avg_song_secs,

            CASE
                WHEN COALESCE(num_unq, 0) > 0
                THEN COALESCE(total_secs, 0) * 1.0 / COALESCE(num_unq, 0)
                ELSE 0
            END AS secs_per_unique,

            (
                CASE
                    WHEN (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                        + COALESCE(num_985, 0) + COALESCE(num_100, 0)) > 0
                    THEN COALESCE(num_100, 0) * 1.0 /
                        (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                        + COALESCE(num_985, 0) + COALESCE(num_100, 0))
                    ELSE 0
                END * 0.6
                +
                CASE
                    WHEN (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                        + COALESCE(num_985, 0) + COALESCE(num_100, 0)) > 0
                    THEN COALESCE(num_985, 0) * 1.0 /
                        (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                        + COALESCE(num_985, 0) + COALESCE(num_100, 0))
                    ELSE 0
                END * 0.3
                -
                CASE
                    WHEN (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                        + COALESCE(num_985, 0) + COALESCE(num_100, 0)) > 0
                    THEN COALESCE(num_25, 0) * 1.0 /
                        (COALESCE(num_25, 0) + COALESCE(num_50, 0) + COALESCE(num_75, 0)
                        + COALESCE(num_985, 0) + COALESCE(num_100, 0))
                    ELSE 0
                END * 0.4
            ) AS quality_score
        FROM user_logs_clean;
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "user_logs_features",
        """
        CREATE TABLE user_logs_features AS
        SELECT
            msno,
            COUNT(*) AS log_day_count,
            MIN(log_date) AS first_log_date,
            MAX(log_date) AS last_log_date,

            SUM(songs_played) AS songs_played_sum,
            AVG(songs_played) AS songs_played_avg,
            STDDEV_SAMP(songs_played) AS songs_played_std,

            SUM(total_secs) AS total_secs_sum,
            AVG(total_secs) AS total_secs_avg,
            STDDEV_SAMP(total_secs) AS total_secs_std,

            SUM(num_unq) AS num_unq_sum,
            AVG(num_unq) AS num_unq_avg,

            AVG(completion_rate) AS completion_rate_avg,
            STDDEV_SAMP(completion_rate) AS completion_rate_std,

            AVG(skip_rate) AS skip_rate_avg,
            STDDEV_SAMP(skip_rate) AS skip_rate_std,

            AVG(near_completion_rate) AS near_completion_rate_avg,
            AVG(repeat_ratio) AS repeat_ratio_avg,
            AVG(avg_song_secs) AS avg_song_secs_avg,
            AVG(secs_per_unique) AS secs_per_unique_avg,
            AVG(quality_score) AS quality_score_avg,

            MAX(total_secs) AS max_total_secs_day,
            MAX(songs_played) AS max_songs_played_day
        FROM user_logs_scored
        GROUP BY msno;
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "latest_user_log_features",
        """
        CREATE TABLE latest_user_log_features AS
        WITH latest_packed AS (
            SELECT
                msno,
                MAX(log_date) AS latest_log_date,
                arg_max(
                    struct_pack(
                        songs_played := songs_played,
                        total_secs := total_secs,
                        num_unq := num_unq,
                        completion_rate := completion_rate,
                        skip_rate := skip_rate,
                        near_completion_rate := near_completion_rate,
                        repeat_ratio := repeat_ratio,
                        avg_song_secs := avg_song_secs,
                        secs_per_unique := secs_per_unique,
                        quality_score := quality_score
                    ),
                    log_date
                ) AS latest_row
            FROM user_logs_scored
            GROUP BY msno
        )
        SELECT
            msno,
            latest_log_date,
            latest_row.songs_played AS latest_songs_played,
            latest_row.total_secs AS latest_total_secs,
            latest_row.num_unq AS latest_num_unq,
            latest_row.completion_rate AS latest_completion_rate,
            latest_row.skip_rate AS latest_skip_rate,
            latest_row.near_completion_rate AS latest_near_completion_rate,
            latest_row.repeat_ratio AS latest_repeat_ratio,
            latest_row.avg_song_secs AS latest_avg_song_secs,
            latest_row.secs_per_unique AS latest_secs_per_unique,
            latest_row.quality_score AS latest_quality_score
        FROM latest_packed;
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "anchors",
        """
        CREATE TABLE anchors AS
        SELECT
            GREATEST(
                COALESCE((SELECT MAX(last_log_date) FROM user_logs_features), DATE '1970-01-01'),
                COALESCE((SELECT MAX(last_transaction_date) FROM transactions_features), DATE '1970-01-01')
            ) AS feature_anchor_date;
        """,
        reuse_existing=reuse_existing,
    )

    _run_stage(
        con,
        "base_features",
        """
        CREATE TABLE base_features AS
        SELECT
            t.msno,
            t.is_churn,

            m.city,
            m.age,
            m.age_missing_or_invalid,
            m.gender_male,
            m.gender_female,
            m.gender_missing,
            m.registered_via,
            m.registration_init_date,

            tx.txn_count,
            tx.first_transaction_date,
            tx.last_transaction_date,
            tx.max_membership_expire_date,
            tx.avg_payment_plan_days,
            tx.max_payment_plan_days,
            tx.total_list_price,
            tx.total_amount_paid,
            tx.avg_amount_paid,
            tx.max_amount_paid,
            tx.auto_renew_rate,
            tx.cancel_rate,
            tx.cancel_count,
            tx.avg_paid_to_list_ratio,
            tx.avg_days_between_transactions,
            tx.avg_gap_from_prev_expire_to_txn,
            tx.post_expiry_renewal_count,
            tx.early_renewal_count,

            ltx.latest_payment_method_id,
            ltx.latest_payment_plan_days,
            ltx.latest_plan_list_price,
            ltx.latest_actual_amount_paid,
            ltx.latest_is_auto_renew,
            ltx.latest_is_cancel,
            ltx.latest_transaction_date,
            ltx.latest_membership_expire_date,

            lg.log_day_count,
            lg.first_log_date,
            lg.last_log_date,
            lg.songs_played_sum,
            lg.songs_played_avg,
            COALESCE(lg.songs_played_std, 0) AS songs_played_std,
            lg.total_secs_sum,
            lg.total_secs_avg,
            COALESCE(lg.total_secs_std, 0) AS total_secs_std,
            lg.num_unq_sum,
            lg.num_unq_avg,
            lg.completion_rate_avg,
            COALESCE(lg.completion_rate_std, 0) AS completion_rate_std,
            lg.skip_rate_avg,
            COALESCE(lg.skip_rate_std, 0) AS skip_rate_std,
            lg.near_completion_rate_avg,
            lg.repeat_ratio_avg,
            lg.avg_song_secs_avg,
            lg.secs_per_unique_avg,
            lg.quality_score_avg,
            lg.max_total_secs_day,
            lg.max_songs_played_day,

            llg.latest_log_date,
            llg.latest_songs_played,
            llg.latest_total_secs,
            llg.latest_num_unq,
            llg.latest_completion_rate,
            llg.latest_skip_rate,
            llg.latest_near_completion_rate,
            llg.latest_repeat_ratio,
            llg.latest_avg_song_secs,
            llg.latest_secs_per_unique,
            llg.latest_quality_score,

            DATE_DIFF('day', lg.last_log_date, a.feature_anchor_date) AS days_since_last_log,
            DATE_DIFF('day', tx.last_transaction_date, a.feature_anchor_date) AS days_since_last_transaction,
            DATE_DIFF('day', a.feature_anchor_date, tx.max_membership_expire_date) AS days_until_membership_expire,

            DATE_DIFF('day', m.registration_init_date, a.feature_anchor_date) AS account_age_days,
            DATE_DIFF('day', tx.first_transaction_date, tx.last_transaction_date) AS subscription_history_days,
            DATE_DIFF('day', lg.first_log_date, lg.last_log_date) + 1 AS observed_log_span_days,

            COALESCE(llg.latest_total_secs, 0) - COALESCE(lg.total_secs_avg, 0) AS latest_vs_avg_total_secs_delta,
            COALESCE(llg.latest_songs_played, 0) - COALESCE(lg.songs_played_avg, 0) AS latest_vs_avg_songs_delta,
            COALESCE(llg.latest_completion_rate, 0) - COALESCE(lg.completion_rate_avg, 0) AS latest_vs_avg_completion_delta,
            COALESCE(llg.latest_skip_rate, 0) - COALESCE(lg.skip_rate_avg, 0) AS latest_vs_avg_skip_delta,
            COALESCE(llg.latest_quality_score, 0) - COALESCE(lg.quality_score_avg, 0) AS latest_vs_avg_quality_delta,

            CASE
                WHEN lg.songs_played_avg > 0 THEN llg.latest_songs_played / lg.songs_played_avg
                ELSE 0
            END AS latest_to_avg_songs_ratio,

            CASE
                WHEN lg.total_secs_avg > 0 THEN llg.latest_total_secs / lg.total_secs_avg
                ELSE 0
            END AS latest_to_avg_total_secs_ratio,

            CASE
                WHEN tx.txn_count > 0 THEN tx.total_amount_paid / tx.txn_count
                ELSE 0
            END AS amount_paid_per_txn,

            CASE
                WHEN lg.log_day_count > 0 THEN lg.total_secs_sum / lg.log_day_count
                ELSE 0
            END AS avg_total_secs_per_logged_day,

            CASE
                WHEN lg.log_day_count > 0 THEN lg.songs_played_sum / lg.log_day_count
                ELSE 0
            END AS avg_songs_per_logged_day,

            CASE
                WHEN tx.max_membership_expire_date >= a.feature_anchor_date THEN 1
                ELSE 0
            END AS membership_active_flag,

            CASE WHEN COALESCE(ltx.latest_is_cancel, 0) = 1 THEN 1 ELSE 0 END AS latest_cancel_flag,
            CASE WHEN COALESCE(ltx.latest_is_auto_renew, 0) = 1 THEN 1 ELSE 0 END AS latest_auto_renew_flag,

            CASE WHEN m.msno IS NOT NULL THEN 1 ELSE 0 END AS has_member_record,
            CASE WHEN tx.msno IS NOT NULL THEN 1 ELSE 0 END AS has_transaction_record,
            CASE WHEN lg.msno IS NOT NULL THEN 1 ELSE 0 END AS has_log_record
        FROM train_base t
        LEFT JOIN members_base m ON t.msno = m.msno
        LEFT JOIN transactions_features tx ON t.msno = tx.msno
        LEFT JOIN latest_transaction_features ltx ON t.msno = ltx.msno
        LEFT JOIN user_logs_features lg ON t.msno = lg.msno
        LEFT JOIN latest_user_log_features llg ON t.msno = llg.msno
        CROSS JOIN anchors a;
        """,
        reuse_existing=reuse_existing,
    )

    if base_output_path is not None and (not base_output_path.exists() or not reuse_existing):
        print(f"Saving base features to {base_output_path}...")
        con.execute(f"COPY base_features TO '{_sql_path(base_output_path)}' (FORMAT PARQUET);")

    print("Loading base features into pandas for capped columns...")
    if base_output_path is not None and base_output_path.exists():
        base_df = pd.read_parquet(base_output_path)
    else:
        base_df = con.execute("SELECT * FROM base_features").df()

    con.close()

    if add_capped_features:
        for col in CAPPED_COLUMNS:
            if col in base_df.columns:
                series = pd.to_numeric(base_df[col], errors="coerce")
                lower = series.quantile(0.01)
                upper = series.quantile(0.99)
                base_df[f"{col}_capped"] = series.clip(lower, upper)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    base_df.to_parquet(output_path, index=False)
    print(f"Saved cleaned feature table to: {output_path}")

    if split_test:
        if test_cutoff is None:
            raise ValueError("split_test=True requires test_cutoff (YYYY-MM-DD).")
        if test_output_path is None:
            raise ValueError("split_test=True requires test_output_path.")
        if test_date_col not in base_df.columns:
            raise ValueError(f"split_test=True requires date column '{test_date_col}' in the feature table.")

        test_cut = pd.to_datetime(test_cutoff)
        date_series = pd.to_datetime(base_df[test_date_col], errors="coerce")
        test_df = base_df[date_series > test_cut].copy()
        train_dev_df = base_df[date_series <= test_cut].copy()

        test_out = Path(test_output_path)
        test_out.parent.mkdir(parents=True, exist_ok=True)
        test_df.to_parquet(test_out, index=False)

        train_dev_out = Path(output_path).with_name(Path(output_path).stem + "_train_dev.parquet")
        train_dev_df.to_parquet(train_dev_out, index=False)

        print(f"Split by {test_date_col} > {test_cutoff}: train/dev={len(train_dev_df)}, test={len(test_df)}")
        print(f"Wrote train/dev to: {train_dev_out}")
        print(f"Wrote test to: {test_out}")


def build_kkbox_feature_frame(
    df: pd.DataFrame,
    target_col: str = DEFAULT_TARGET_COL,
    id_col: str = DEFAULT_ID_COL,
) -> FeatureArtifacts:
    if target_col not in df.columns:
        raise ValueError(f"Expected target column '{target_col}' in KKBox dataset.")

    y = df[target_col].astype(int).copy()

    drop_cols: List[str] = [target_col]
    if id_col in df.columns:
        drop_cols.append(id_col)

    # Drop raw date columns: keep only engineered recency/tenure features
    drop_cols.extend([c for c in RAW_DATE_COLUMNS if c in df.columns])

    # Prefer capped versions over raw heavy-tailed versions when both exist
    for raw_col, capped_col in CAPPED_PREFERRED_PAIRS.items():
        if raw_col in df.columns and capped_col in df.columns:
            drop_cols.append(raw_col)

    X = df.drop(columns=[c for c in drop_cols if c in df.columns]).copy()

    numeric_cols: List[str] = []
    categorical_cols: List[str] = []

    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]) and not pd.api.types.is_bool_dtype(X[col]):
            numeric_cols.append(col)
        else:
            X[col] = X[col].astype("string").fillna("__missing__")
            categorical_cols.append(col)

    return FeatureArtifacts(X=X, y=y, numeric_cols=numeric_cols, categorical_cols=categorical_cols)


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


if __name__ == "__main__":
    build_feature_table(
        train_path=TRAIN_PATH,
        members_path=MEMBERS_PATH,
        transactions_path=TRANSACTIONS_PATH,
        user_logs_path=USER_LOGS_PATH,
        output_path=OUTPUT_PATH,
        base_output_path=BASE_OUTPUT_PATH,
        add_capped_features=True,
        reuse_existing=True,
        force_recap_only=False,
        split_test=False,
        test_date_col="latest_transaction_date",
        test_cutoff="2017-02-28",
        test_output_path=OUTPUT_DIR / "kkbox_test.parquet",
    )
