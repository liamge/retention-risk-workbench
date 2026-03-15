from __future__ import annotations

import logging
from typing import Dict

from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

try:
    from xgboost import XGBClassifier

    XGBOOST_AVAILABLE = True
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning("XGBoost unavailable: %s", e)
    XGBOOST_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier

    LGBM_AVAILABLE = True
except Exception as e:  # pragma: no cover - optional dependency
    logger.warning("LightGBM unavailable: %s", e)
    LGBM_AVAILABLE = False


def get_candidate_models(cfg: Dict, class_ratio: float) -> Dict[str, object]:
    """Return instantiated candidate estimators configured from YAML."""
    candidates: Dict[str, object] = {
        "logistic_regression": LogisticRegression(
            max_iter=cfg["models"]["logistic_regression"].get("max_iter", 2000)
        )
    }

    if XGBOOST_AVAILABLE:
        candidates["xgboost"] = XGBClassifier(
            objective="binary:logistic",
            eval_metric="auc",
            n_estimators=cfg["models"]["xgboost"].get("n_estimators", 400),
            learning_rate=cfg["models"]["xgboost"].get("learning_rate", 0.05),
            max_depth=cfg["models"]["xgboost"].get("max_depth", 4),
            subsample=cfg["models"]["xgboost"].get("subsample", 0.9),
            colsample_bytree=cfg["models"]["xgboost"].get("colsample_bytree", 0.9),
            min_child_weight=cfg["models"]["xgboost"].get("min_child_weight", 1),
            reg_lambda=cfg["models"]["xgboost"].get("reg_lambda", 1.0),
            scale_pos_weight=class_ratio,
            random_state=cfg["split"].get("random_state", 42),
        )

    if LGBM_AVAILABLE:
        candidates["lightgbm"] = LGBMClassifier(
            objective="binary",
            n_estimators=cfg["models"]["lightgbm"].get("n_estimators", 400),
            learning_rate=cfg["models"]["lightgbm"].get("learning_rate", 0.05),
            max_depth=cfg["models"]["lightgbm"].get("max_depth", -1),
            num_leaves=cfg["models"]["lightgbm"].get("num_leaves", 31),
            subsample=cfg["models"]["lightgbm"].get("subsample", 0.9),
            colsample_bytree=cfg["models"]["lightgbm"].get("colsample_bytree", 0.9),
            reg_lambda=cfg["models"]["lightgbm"].get("reg_lambda", 1.0),
            min_child_samples=cfg["models"]["lightgbm"].get("min_child_samples", 20),
            class_weight={0: 1.0, 1: class_ratio},
            random_state=cfg["split"].get("random_state", 42),
            verbose=-1,
        )

    return candidates
