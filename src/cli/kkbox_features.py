from __future__ import annotations

"""CLI entrypoint for generating KKBox feature tables."""

import argparse
import logging
import sys
from pathlib import Path

from src.kkbox.engineering import build_feature_table
from src.utils.io import ensure_dir

# Defaults kept here so CLI behavior stays stable even as the compatibility
# shim (src/kkbox_features.py) remains thin.
DATA_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")

TRAIN_PATH = DATA_DIR / "train.csv"
MEMBERS_PATH = DATA_DIR / "members.csv"
TRANSACTIONS_PATH = DATA_DIR / "transactions.csv"
USER_LOGS_PATH = DATA_DIR / "user_logs.csv"

DUCKDB_PATH = OUTPUT_DIR / "kkbox_work.duckdb"
TEMP_DIR = OUTPUT_DIR / "duckdb_tmp"

BASE_OUTPUT_PATH = OUTPUT_DIR / "kkbox_features_base.parquet"
OUTPUT_PATH = OUTPUT_DIR / "kkbox_features_duckdb_cleaned.parquet"
DEFAULT_CONFIG_PATH = Path("configs/kkbox.yaml")

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build KKBox feature tables.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH), help="Path to kkbox config yaml.")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH), help="Clean feature parquet path.")
    parser.add_argument("--base-output", type=str, default=str(BASE_OUTPUT_PATH), help="Base features parquet path.")
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Optional reservoir sample fraction (0,1] for test mode to avoid loading full dataset.",
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Convenience flag: set sample_frac=0.02 and reuse_existing=False to force a small rebuild.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    """CLI entrypoint for building KKBox feature tables."""
    args = args or parse_args()

    sample_frac = args.sample_frac
    reuse_existing = True
    if args.test_mode:
        sample_frac = sample_frac or 0.02
        reuse_existing = False

    ensure_dir(OUTPUT_DIR)
    ensure_dir(TEMP_DIR)

    try:
        build_feature_table(
            train_path=TRAIN_PATH,
            members_path=MEMBERS_PATH,
            transactions_path=TRANSACTIONS_PATH,
            user_logs_path=USER_LOGS_PATH,
            output_path=Path(args.output),
            base_output_path=Path(args.base_output),
            duckdb_path=DUCKDB_PATH,
            temp_dir=TEMP_DIR,
            sample_frac=sample_frac,
            add_capped_features=True,
            reuse_existing=reuse_existing,
            force_recap_only=False,
            split_test=True,
            test_date_col="latest_transaction_date",
            test_cutoff="2017-02-28",
            test_output_path=OUTPUT_DIR / "kkbox_test.parquet",
            config_path=args.config,
        )
    except Exception:
        logger.exception("Failed to build KKBox feature tables.")
        return 1

    logger.info("KKBox features written to %s", Path(args.output).resolve())
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    sys.exit(main())
