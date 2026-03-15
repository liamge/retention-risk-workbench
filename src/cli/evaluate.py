from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Iterable, Optional, Tuple
import warnings

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay

from src.features import build_feature_frame
from src.kkbox_features import build_kkbox_feature_frame
from src.utils.io import ensure_dir, load_config, read_table
from src.utils.splits import get_test_split

logger = logging.getLogger(__name__)


TEST_SPLIT_CANDIDATES = (
    "splits/test_ids.parquet",
    "splits/test_ids.csv",
    "splits/test_rows.parquet",
    "splits/test_rows.csv",
    "splits/test_indices.parquet",
    "splits/test_indices.csv",
    "test_set.parquet",
    "test_set.csv",
)


def _load_model_metadata(artifact_dir: Path) -> dict:
    metadata_path = artifact_dir / "model_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"Expected model metadata at {metadata_path}; run training before evaluation."
        )
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_id_column(df: pd.DataFrame, dataset_type: str) -> Optional[str]:
    candidates = [
        "customerID",
        "customer_id",
        "id",
        "ID",
    ]

    if dataset_type == "kkbox":
        candidates = ["msno", "member_id", "user_id", *candidates]

    for col in candidates:
        if col in df.columns:
            return col
    return None


def _first_present(columns: Iterable[str], candidates: Iterable[str]) -> Optional[str]:
    for cand in candidates:
        if cand in columns:
            return cand
    return None


def _load_saved_split_table(artifact_dir: Path) -> tuple[Optional[Path], Optional[pd.DataFrame]]:
    for rel_path in TEST_SPLIT_CANDIDATES:
        candidate = artifact_dir / rel_path
        if candidate.exists():
            try:
                return candidate, read_table(candidate)
            except Exception as exc:  # pragma: no cover - defensive
                warnings.warn(f"Failed to read saved split at {candidate}: {exc}")
    return None, None


def _filter_saved_table(saved_df: pd.DataFrame) -> pd.DataFrame:
    for split_col in ("split", "partition", "dataset"):
        if split_col in saved_df.columns:
            mask = saved_df[split_col].astype(str).str.lower().str.contains("test")
            return saved_df.loc[mask].copy()

    if "is_test" in saved_df.columns:
        mask = saved_df["is_test"].astype(bool)
        return saved_df.loc[mask].copy()

    # If no split indicator is present, assume the file is already the test slice.
    return saved_df.copy()


def _test_split_from_saved(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    saved_df: pd.DataFrame,
    id_col: Optional[str],
    target_col: str,
    source_path: Path,
) -> Tuple[pd.DataFrame, pd.Series]:
    filtered = _filter_saved_table(saved_df)

    # If the saved artifact already contains engineered features, ensure they align exactly.
    if set(X.columns).issubset(filtered.columns):
        y_source = None
        for cand in (target_col, "target", "label", "y"):
            if cand in filtered.columns:
                y_source = filtered[cand]
                break
        if y_source is None:
            raise ValueError(
                f"Saved split at {source_path} includes features but no target column ({target_col})."
            )
        return filtered[X.columns].copy(), y_source.astype(y.dtype).copy()

    id_candidates = [c for c in (id_col, "row_id", "index", "idx", "id") if c]
    saved_id_col = _first_present(filtered.columns, id_candidates)

    if saved_id_col:
        saved_ids = filtered[saved_id_col].astype(str)
        if X.index.astype(str).isin(saved_ids).sum() == 0:
            raise ValueError(
                "Saved split IDs did not match any rows in the current dataset. "
                "Confirm the same data file and split artifacts are used."
            )
        mask = X.index.astype(str).isin(set(saved_ids))
        return X.loc[mask].copy(), y.loc[mask].copy()

    raise ValueError(
        f"Could not align saved split from {source_path} — no usable ID columns found."
    )


def _validate_features(feature_cols: Iterable[str], metadata: dict) -> None:
    expected_cols = metadata.get("feature_columns")
    if not expected_cols:
        return

    missing = set(expected_cols) - set(feature_cols)
    extra = set(feature_cols) - set(expected_cols)
    if missing or extra:
        raise ValueError(
            "Feature column mismatch with training metadata. "
            f"Missing: {sorted(missing)} | Unexpected: {sorted(extra)}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate saved churn model and write figures/reports.")
    parser.add_argument("--config", type=str, default="configs/base.yaml")
    parser.add_argument("--data-path", type=str, default=None, help="Optional override for data path")
    parser.add_argument("--artifact-dir", type=str, default=None, help="Optional override for artifact dir")
    parser.add_argument("--report-dir", type=str, default="reports")
    return parser.parse_args()


def main(args: argparse.Namespace | None = None) -> int:
    args = args or parse_args()
    cfg = load_config(args.config)

    data_cfg = cfg["data"]
    data_path = args.data_path or data_cfg.get("raw_path") or data_cfg.get("feature_path")
    if not data_path:
        raise KeyError("Config is missing data.raw_path (or data.feature_path for KKBox).")
    dataset_type = cfg["data"].get("dataset_type", "telco").lower()
    target_col = cfg["data"].get("target_col", "Churn" if dataset_type == "telco" else "is_churn")
    split_cfg = cfg["split"]
    train_size = split_cfg.get("train_size")
    dev_size = split_cfg.get("dev_size")
    test_size = split_cfg.get("test_size")
    split_strategy = split_cfg.get("strategy", "time" if dataset_type == "kkbox" else "random")
    date_col_default = "latest_transaction_date" if dataset_type == "kkbox" else "transaction_date"
    date_col = split_cfg.get("date_col", date_col_default)
    artifact_dir = Path(args.artifact_dir or cfg["artifacts"]["dir"])
    report_dir = ensure_dir(args.report_dir)
    figure_dir = ensure_dir(report_dir / "figures")

    metadata = _load_model_metadata(artifact_dir)
    trained_dataset = metadata.get("dataset_type", dataset_type).lower()
    if trained_dataset != dataset_type:
        raise ValueError(
            f"Dataset type mismatch: training='{trained_dataset}' vs config='{dataset_type}'. "
            "Use the config that matches the trained model."
        )

    trained_source = metadata.get("data_source")
    if trained_source and Path(trained_source).resolve() != Path(data_path).resolve():
        raise ValueError(
            "Data source does not match training metadata. "
            f"Training used: {trained_source} | Evaluation requested: {data_path}"
        )

    df = read_table(data_path)
    id_col = _find_id_column(df, dataset_type)
    if id_col:
        df = df.set_index(id_col, drop=False)

    if dataset_type == "kkbox":
        fa = build_kkbox_feature_frame(df, target_col=target_col)
    else:
        fa = build_feature_frame(df)

    X, y = fa.X, fa.y
    _validate_features(X.columns, metadata)

    split_source = "recomputed from config"
    saved_split_path, saved_split_df = _load_saved_split_table(artifact_dir)
    if saved_split_df is not None:
        X_test, y_test = _test_split_from_saved(
            X,
            y,
            saved_df=saved_split_df,
            id_col=id_col,
            target_col=target_col,
            source_path=saved_split_path or artifact_dir,
        )
        split_source = f"saved split at {saved_split_path}"
    else:
        time_split_meta = metadata.get("time_split") or {}
        if split_strategy == "time" and time_split_meta:
            # Enforce that evaluation uses the same temporal boundaries as training.
            meta_dev_end = time_split_meta.get("dev_end")
            meta_date_col = time_split_meta.get("date_col") or date_col

            if meta_date_col and meta_date_col != date_col:
                raise ValueError(
                    "Configured date_col differs from training metadata; refusing to recompute split."
                )

            if meta_dev_end and split_cfg.get("dev_end") and str(split_cfg["dev_end"]) != str(meta_dev_end):
                raise ValueError(
                    "Configured dev_end differs from training metadata; refusing to recompute split."
                )

            date_col = meta_date_col
            dev_end = meta_dev_end or split_cfg.get("dev_end")
        else:
            dev_end = split_cfg.get("dev_end")

        warnings.warn(
            "No saved test split artifacts found; recomputing from split config. "
            "If raw row ordering differs from training, results may drift.",
            UserWarning,
        )

        X_test, y_test = get_test_split(
            X,
            y,
            dataset_name=dataset_type,
            split_strategy=split_strategy,
            random_state=split_cfg.get("random_state", 42),
            train_size=train_size,
            dev_size=dev_size,
            test_size=test_size,
            date_col=date_col,
            dev_end=dev_end,
        )

    if X_test.empty:
        raise ValueError("Test split is empty after alignment; check split artifacts and config.")

    model = joblib.load(artifact_dir / "model.pkl")
    threshold = float(metadata["threshold"])

    y_score = model.predict_proba(X_test)[:, 1]
    y_pred = (y_score >= threshold).astype(int)

    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_score, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(figure_dir / "roc_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_score, ax=ax)
    ax.set_title("Precision-Recall Curve")
    fig.tight_layout()
    fig.savefig(figure_dir / "precision_recall_curve.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title(f"Confusion Matrix @ threshold={threshold:.3f}")
    fig.tight_layout()
    fig.savefig(figure_dir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    scored = X_test.copy()
    if id_col and id_col not in scored.columns:
        scored.insert(0, id_col, scored.index)
    scored["churn_probability"] = y_score
    scored["predicted_churn"] = y_pred
    scored[target_col] = y_test.values

    if "MonthlyCharges" in X_test.columns:
        scored["estimated_monthly_revenue_at_risk"] = X_test["MonthlyCharges"].values * y_score

    scored.to_csv(report_dir / "test_set_scored.csv", index=False)

    total_high_risk = int((y_pred == 1).sum())
    total_at_risk = float(scored.get("estimated_monthly_revenue_at_risk", pd.Series(dtype=float)).sum())

    summary = f"""# Executive Summary

## Model
- Champion model: {metadata['champion_model']}
- Classification threshold: {threshold:.3f}

## Test Set Snapshot
- Customers scored: {len(scored)}
- Predicted high-risk customers: {total_high_risk}
- Estimated monthly revenue at risk: ${total_at_risk:,.2f}

## Notes
- This report uses the saved champion pipeline and the held-out test split.
- Split source: {split_source}
- Revenue at risk is a proxy based on churn probability × monthly charges.
- Next iteration: add calibration plots, driver analysis, and monitoring/drift reporting.
"""

    with open(report_dir / "executive_summary.md", "w", encoding="utf-8") as f:
        f.write(summary)

    logger.info("Test split source: %s", split_source)
    logger.info("Wrote figures to %s", figure_dir.resolve())
    logger.info("Wrote summary to %s", (report_dir / "executive_summary.md").resolve())
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Evaluation run failed.")
        sys.exit(1)
