from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def choose_threshold(y_true, y_proba, metric: str = "f1", step: float = 0.01) -> float:
    """
    Pick a classification threshold from probabilities.
    Currently supports F1.
    """
    if metric != "f1":
        raise ValueError(f"Unsupported metric: {metric}")

    thresholds = np.arange(step, 1.0, step)
    scores = []

    for t in thresholds:
        preds = (y_proba >= t).astype(int)
        scores.append(f1_score(y_true, preds, zero_division=0))

    best_idx = int(np.argmax(scores))
    return float(thresholds[best_idx])