from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import json
import pandas as pd
from pandas.api import types as ptypes
import yaml


def ensure_dir(path: str | Path) -> Path:
    """Create a directory (and parents) if it doesn't exist and return the Path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    if path.suffix in {".yaml", ".yml"}:
        with path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if path.suffix == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)

    raise ValueError(f"Unsupported config format: {path.suffix}")


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        df = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported table format: {path.suffix}")

    # Normalize string-like columns to plain Python strings to avoid Arrow LargeUtf8 issues.
    for col in df.columns:
        series = df[col]
        if ptypes.is_string_dtype(series) or ptypes.is_object_dtype(series) or ptypes.is_categorical_dtype(series):
            # Force Python string objects rather than Arrow/pyarrow-backed dtypes
            df[col] = series.astype(str)

    return df
