"""Compatibility facade for KKBox feature engineering.

This module keeps legacy imports working (`from src.kkbox_features import ...`).
Programmatic callers should use `src.kkbox.engineering`, and CLI execution now
lives in `src.cli.kkbox_features`.
"""

from src.kkbox.engineering import (
    DEFAULT_ID_COL,
    DEFAULT_TARGET_COL,
    build_feature_table,
    build_kkbox_feature_frame,
    make_kkbox_preprocessor,
)

__all__ = [
    "DEFAULT_ID_COL",
    "DEFAULT_TARGET_COL",
    "build_feature_table",
    "build_kkbox_feature_frame",
    "make_kkbox_preprocessor",
]


if __name__ == "__main__":  # pragma: no cover - legacy script shim
    import sys
    from src.cli.kkbox_features import main

    sys.exit(main())
