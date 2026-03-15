from pathlib import Path

import pandas as pd
import pytest

from src.evaluate import _test_split_from_saved, _validate_features


def test_test_split_from_saved_uses_row_ids():
    X = pd.DataFrame({"f": [1, 2, 3]}, index=pd.Index(["a", "b", "c"], name="customerID"))
    y = pd.Series([0, 1, 0], index=X.index, name="ChurnFlag")

    saved_df = pd.DataFrame({"row_id": ["a", "c"], "split": ["test", "test"]})

    X_test, y_test = _test_split_from_saved(
        X,
        y,
        saved_df=saved_df,
        id_col="customerID",
        target_col="ChurnFlag",
        source_path=Path("dummy"),
    )

    assert list(X_test.index) == ["a", "c"]
    assert list(y_test.index) == ["a", "c"]
    assert list(X_test["f"]) == [1, 3]


def test_test_split_from_saved_raises_on_id_mismatch():
    X = pd.DataFrame({"f": [1, 2]}, index=pd.Index(["a", "b"], name="customerID"))
    y = pd.Series([0, 1], index=X.index, name="ChurnFlag")
    saved_df = pd.DataFrame({"row_id": ["x"]})

    with pytest.raises(ValueError):
        _test_split_from_saved(
            X,
            y,
            saved_df=saved_df,
            id_col="customerID",
            target_col="ChurnFlag",
            source_path=Path("dummy"),
        )


def test_validate_features_detects_mismatch():
    with pytest.raises(ValueError):
        _validate_features(["a", "b"], {"feature_columns": ["a", "c"]})
