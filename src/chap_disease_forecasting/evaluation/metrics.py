"""Metrics with explicit handling of zeros and probabilistic inputs."""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np


def _aligned_finite(
    actual: Iterable[float], predicted: Iterable[float]
) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    if y.shape != p.shape:
        raise ValueError("actual and predicted must have equal shape")
    mask = np.isfinite(y) & np.isfinite(p)
    if not mask.any():
        raise ValueError("no aligned finite observations")
    return y[mask], p[mask]


def point_metrics(actual: Iterable[float], predicted: Iterable[float]) -> dict[str, float | int]:
    """Return MAE/RMSE and MAPE excluding zero-valued actuals.

    ``mape_n`` reports the denominator count so a MAPE computed on very few
    observations cannot be mistaken for a complete evaluation.
    """

    y, p = _aligned_finite(actual, predicted)
    error = p - y
    nonzero = y != 0
    mape = (
        float(np.mean(np.abs(error[nonzero] / y[nonzero])) * 100.0) if nonzero.any() else math.nan
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mape": mape,
        "n": int(len(y)),
        "mape_n": int(nonzero.sum()),
    }


def empirical_crps(actual: Iterable[float], samples: np.ndarray) -> float:
    """Mean CRPS for rows of genuine predictive samples."""

    y = np.asarray(list(actual), dtype=float)
    draws = np.asarray(samples, dtype=float)
    if draws.ndim != 2 or draws.shape[0] != len(y) or draws.shape[1] < 2:
        raise ValueError("samples must be (n_observations, n_samples>=2)")
    if not np.isfinite(y).all() or not np.isfinite(draws).all():
        raise ValueError("CRPS inputs must be finite")
    first = np.mean(np.abs(draws - y[:, None]), axis=1)
    sorted_draws = np.sort(draws, axis=1)
    n = draws.shape[1]
    coefficients = 2 * np.arange(1, n + 1) - n - 1
    pairwise_half = np.sum(sorted_draws * coefficients, axis=1) / (n * n)
    return float(np.mean(first - pairwise_half))


def interval_coverage(
    actual: Iterable[float], lower: Iterable[float], upper: Iterable[float]
) -> float:
    y, lo = _aligned_finite(actual, lower)
    y2, hi = _aligned_finite(actual, upper)
    if not np.array_equal(y, y2):
        raise ValueError("interval arrays have inconsistent missingness")
    if (lo > hi).any():
        raise ValueError("lower interval bound exceeds upper bound")
    return float(np.mean((y >= lo) & (y <= hi)))


def interval_score(
    actual: Iterable[float],
    lower: Iterable[float],
    upper: Iterable[float],
    alpha: float = 0.1,
) -> float:
    if not 0 < alpha < 1:
        raise ValueError("alpha must lie between 0 and 1")
    y = np.asarray(list(actual), dtype=float)
    lo = np.asarray(list(lower), dtype=float)
    hi = np.asarray(list(upper), dtype=float)
    if y.shape != lo.shape or y.shape != hi.shape:
        raise ValueError("interval inputs must have equal shape")
    mask = np.isfinite(y) & np.isfinite(lo) & np.isfinite(hi)
    y, lo, hi = y[mask], lo[mask], hi[mask]
    if len(y) == 0 or (lo > hi).any():
        raise ValueError("invalid interval inputs")
    score = hi - lo + (2.0 / alpha) * (lo - y) * (y < lo) + (2.0 / alpha) * (y - hi) * (y > hi)
    return float(np.mean(score))


def alert_metrics(
    actual: Iterable[float], predicted: Iterable[float], threshold: Iterable[float]
) -> dict[str, float | int]:
    y = np.asarray(list(actual), dtype=float)
    p = np.asarray(list(predicted), dtype=float)
    t = np.asarray(list(threshold), dtype=float)
    if y.shape != p.shape or y.shape != t.shape:
        raise ValueError("alert inputs must have equal shape")
    mask = np.isfinite(y) & np.isfinite(p) & np.isfinite(t)
    observed_alert = y[mask] >= t[mask]
    predicted_alert = p[mask] >= t[mask]
    true_positive = int(np.sum(observed_alert & predicted_alert))
    false_positive = int(np.sum(~observed_alert & predicted_alert))
    positives = int(np.sum(observed_alert))
    negatives = int(np.sum(~observed_alert))
    return {
        "sensitivity": true_positive / positives if positives else math.nan,
        "false_positive_rate": false_positive / negatives if negatives else math.nan,
        "true_positives": true_positive,
        "false_positives": false_positive,
        "n": int(mask.sum()),
    }
