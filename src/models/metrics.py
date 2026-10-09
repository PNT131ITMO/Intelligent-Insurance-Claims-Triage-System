from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score


METRIC_COLUMNS = ("log_loss", "roc_auc", "average_precision", "brier_score")


def binary_probability_metrics(y_true: Iterable[int], probability: Iterable[float]) -> dict[str, float]:
    y_array = np.asarray(y_true)
    probability_array = np.asarray(probability, dtype=np.float64)
    if y_array.ndim != 1 or probability_array.ndim != 1 or len(y_array) != len(probability_array):
        raise ValueError("Target and probability must be aligned one-dimensional arrays")
    if set(np.unique(y_array)) != {0, 1}:
        raise ValueError("Target must contain both binary classes 0 and 1")
    if not np.isfinite(probability_array).all():
        raise ValueError("Predicted probabilities must be finite")
    if ((probability_array < 0) | (probability_array > 1)).any():
        raise ValueError("Predicted probabilities must be within [0, 1]")
    return {
        "log_loss": float(log_loss(y_array, probability_array, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_array, probability_array)),
        "average_precision": float(average_precision_score(y_array, probability_array)),
        "brier_score": float(brier_score_loss(y_array, probability_array)),
    }


def summarize_fold_metrics(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    required = {"model", "feature_set", "fold", *METRIC_COLUMNS}
    missing = required.difference(fold_metrics.columns)
    if missing:
        raise ValueError(f"Fold metrics are missing columns: {sorted(missing)}")
    summary = fold_metrics.groupby(["model", "feature_set"], sort=False)[list(METRIC_COLUMNS)].agg(["mean", "std"])
    summary.columns = [f"{metric}_{statistic}" for metric, statistic in summary.columns]
    return summary.reset_index()

