from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import joblib
import pandas as pd

from src.utils.io import ensure_dir


def save_candidate_results(results_df: pd.DataFrame, outpath: Path) -> None:
    ensure_dir(outpath.parent)
    results_df.to_csv(outpath, index=False)


def save_metadata(metadata: Dict[str, Any], outpath: Path) -> None:
    ensure_dir(outpath.parent)
    with open(outpath, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)


def save_model(model, outpath: Path) -> None:
    ensure_dir(outpath.parent)
    joblib.dump(model, outpath)


def save_test_artifacts(
    X_test: pd.DataFrame,
    y_test: pd.Series,
    id_col: str | None,
    target_name: str,
    splits_dir: Path,
) -> None:
    ensure_dir(splits_dir)

    test_ids = pd.DataFrame({"row_id": X_test.index})
    if id_col and id_col not in test_ids.columns:
        test_ids[id_col] = test_ids["row_id"]
    test_ids.to_parquet(splits_dir / "test_ids.parquet", index=False)
    test_ids.to_csv(splits_dir / "test_ids.csv", index=False)

    test_full = X_test.copy()
    if id_col and id_col not in test_full.columns:
        test_full.insert(0, id_col, X_test.index)
    test_full[target_name] = y_test.values
    test_full.to_parquet(splits_dir / "test_set.parquet", index=False)
    test_full.to_csv(splits_dir / "test_set.csv", index=False)
