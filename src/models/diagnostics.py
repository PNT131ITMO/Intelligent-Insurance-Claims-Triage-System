from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .metrics import binary_probability_metrics


OOF_COLUMNS = {"ID", "target", "fold", "catboost_probability"}


def validate_oof_predictions(oof: pd.DataFrame) -> None:
    missing = OOF_COLUMNS.difference(oof.columns)
    if missing:
        raise ValueError(f"OOF predictions are missing columns: {sorted(missing)}")
    if oof.empty or oof[list(OOF_COLUMNS)].isna().any().any():
        raise ValueError("OOF predictions must be nonempty and complete")
    if not oof["ID"].is_unique:
        raise ValueError("Each training ID must have exactly one OOF prediction")
    if set(oof["target"].unique()) != {0, 1}:
        raise ValueError("OOF target must contain exactly the binary labels 0 and 1")
    fold = oof["fold"].to_numpy(dtype=np.float64)
    if not np.isfinite(fold).all() or ((fold < 1) | (fold != np.floor(fold))).any():
        raise ValueError("OOF fold numbers must be positive integers")
    probability = oof["catboost_probability"].to_numpy(dtype=np.float64)
    if not np.isfinite(probability).all() or ((probability < 0) | (probability > 1)).any():
        raise ValueError("OOF probabilities must be finite values in [0, 1]")


def _validated_arrays(
    y_true: Iterable[int], probability: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y_true)
    p = np.asarray(probability, dtype=np.float64)
    if y.ndim != 1 or p.ndim != 1 or len(y) != len(p) or len(y) == 0:
        raise ValueError("Target and probability must be nonempty aligned vectors")
    if not np.isin(y, [0, 1]).all():
        raise ValueError("Target values must be 0 or 1")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Probabilities must be finite values in [0, 1]")
    return y.astype(np.int8, copy=False), p


def calibration_bins(
    y_true: Iterable[int], probability: Iterable[float], *, n_bins: int = 10
) -> pd.DataFrame:
    """Return fixed-width reliability bins, including bins with no observations."""
    if n_bins < 2:
        raise ValueError("n_bins must be at least 2")
    y, p = _validated_arrays(y_true, probability)
    bin_index = np.minimum(np.floor(p * n_bins).astype(np.int64), n_bins - 1)
    count = np.bincount(bin_index, minlength=n_bins)
    positives = np.bincount(bin_index, weights=y, minlength=n_bins)
    probability_sum = np.bincount(bin_index, weights=p, minlength=n_bins)
    mean_probability = np.divide(
        probability_sum, count, out=np.full(n_bins, np.nan), where=count > 0
    )
    observed_rate = np.divide(
        positives, count, out=np.full(n_bins, np.nan), where=count > 0
    )
    return pd.DataFrame({
        "bin": np.arange(n_bins),
        "lower": np.arange(n_bins) / n_bins,
        "upper": (np.arange(n_bins) + 1) / n_bins,
        "count": count,
        "positive_count": positives.astype(np.int64),
        "mean_probability": mean_probability,
        "observed_rate": observed_rate,
        "absolute_gap": np.abs(observed_rate - mean_probability),
    })


def expected_calibration_error(bins: pd.DataFrame) -> float:
    required = {"count", "absolute_gap"}
    if not required.issubset(bins.columns) or bins.empty or bins["count"].sum() <= 0:
        raise ValueError("Calibration bins must contain nonzero counts and absolute gaps")
    if bins["count"].isna().any() or (bins["count"] < 0).any():
        raise ValueError("Calibration bin counts must be nonnegative")
    if bins.loc[bins["count"].gt(0), "absolute_gap"].isna().any():
        raise ValueError("Occupied calibration bins must have an absolute gap")
    return float((bins["count"] * bins["absolute_gap"].fillna(0)).sum() / bins["count"].sum())


def threshold_summary(
    y_true: Iterable[int], probability: Iterable[float], thresholds: Iterable[float]
) -> pd.DataFrame:
    y, p = _validated_arrays(y_true, probability)
    rows: list[dict[str, float | int]] = []
    for threshold in thresholds:
        threshold = float(threshold)
        if not np.isfinite(threshold) or not 0 <= threshold <= 1:
            raise ValueError("Thresholds must be finite values in [0, 1]")
        predicted_positive = p >= threshold
        tp = int(np.count_nonzero(predicted_positive & (y == 1)))
        fp = int(np.count_nonzero(predicted_positive & (y == 0)))
        tn = int(np.count_nonzero(~predicted_positive & (y == 0)))
        fn = int(np.count_nonzero(~predicted_positive & (y == 1)))
        rows.append({
            "threshold": threshold,
            "predicted_positive": tp + fp,
            "predicted_positive_share": (tp + fp) / len(y),
            "true_positive": tp,
            "false_positive": fp,
            "true_negative": tn,
            "false_negative": fn,
            "precision_class_1": tp / (tp + fp) if tp + fp else np.nan,
            "recall_class_1": tp / (tp + fn) if tp + fn else np.nan,
            "false_positive_rate": fp / (fp + tn) if fp + tn else np.nan,
            "specificity": tn / (tn + fp) if tn + fp else np.nan,
        })
    if not rows:
        raise ValueError("At least one threshold is required")
    return pd.DataFrame(rows)


def segment_summary(
    oof: pd.DataFrame, segment: pd.Series, *, minimum_rows: int = 100
) -> pd.DataFrame:
    validate_oof_predictions(oof)
    if minimum_rows < 1 or not segment.index.equals(oof.index) or segment.isna().any():
        raise ValueError("Segment labels must be aligned, complete, and have a positive minimum")
    rows: list[dict[str, float | int | str]] = []
    for label, positions in segment.groupby(segment, sort=False).indices.items():
        group = oof.iloc[positions]
        if len(group) < minimum_rows or group["target"].nunique() != 2:
            continue
        metrics = binary_probability_metrics(group["target"], group["catboost_probability"])
        rows.append({
            "segment": str(label),
            "rows": len(group),
            "observed_rate": float(group["target"].mean()),
            "mean_probability": float(group["catboost_probability"].mean()),
            **metrics,
        })
    return pd.DataFrame(rows)


def highest_loss_cases(oof: pd.DataFrame, *, top_n: int = 50) -> pd.DataFrame:
    validate_oof_predictions(oof)
    if top_n < 1:
        raise ValueError("top_n must be positive")
    y, p = _validated_arrays(oof["target"], oof["catboost_probability"])
    clipped = np.clip(p, np.finfo(np.float64).eps, 1 - np.finfo(np.float64).eps)
    loss = -(y * np.log(clipped) + (1 - y) * np.log1p(-clipped))
    worst = np.argsort(-loss, kind="stable")[:top_n]
    output = oof.iloc[worst][["ID", "target", "fold", "catboost_probability"]].copy()
    output["row_log_loss"] = loss[worst]
    return output.reset_index(drop=True)
