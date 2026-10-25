from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from chap_disease_forecasting.evaluation import empirical_crps, interval_coverage, interval_score
from chap_disease_forecasting.forecasting import (
    HorizonResidualSampler,
    InsufficientResidualEvidence,
)


def _backtest() -> pd.DataFrame:
    rows = []
    for horizon in (1, 2, 3):
        for index in range(30):
            rows.append(
                {
                    "actual": 10 + index % 7,
                    "prediction": 9 + (index * 3) % 8,
                    "horizon": horizon,
                    "forecast_origin": pd.Timestamp("2025-01-01") + pd.Timedelta(weeks=index),
                    "is_out_of_sample": True,
                }
            )
    return pd.DataFrame(rows)


def test_residual_samples_are_reproducible_non_identical_and_labelled_honestly() -> None:
    points = pd.DataFrame(
        {"location": ["A", "A", "A"], "horizon": [1, 2, 3], "prediction": [10, 11, 12]}
    )
    first = HorizonResidualSampler(n_samples=50, seed=9).fit(_backtest()).add_samples(points)
    second = HorizonResidualSampler(n_samples=50, seed=9).fit(_backtest()).add_samples(points)
    sample_columns = [f"sample_{index}" for index in range(50)]
    assert np.array_equal(first[sample_columns].to_numpy(), second[sample_columns].to_numpy())
    assert np.unique(first.loc[0, sample_columns].to_numpy()).size > 1
    assert first["probabilistic_method"].eq("empirical_oos_log_residual").all()
    assert not first["probabilistic_calibrated"].any()
    actual = np.array([11, 10, 14], dtype=float)
    samples = first[sample_columns].to_numpy(dtype=float)
    assert empirical_crps(actual, samples) >= 0
    lower = np.quantile(samples, 0.05, axis=1)
    upper = np.quantile(samples, 0.95, axis=1)
    assert 0 <= interval_coverage(actual, lower, upper) <= 1
    assert interval_score(actual, lower, upper) >= 0


def test_in_sample_or_insufficient_residuals_are_rejected() -> None:
    invalid = _backtest()
    invalid.loc[0, "is_out_of_sample"] = False
    with pytest.raises(ValueError, match="out-of-sample"):
        HorizonResidualSampler().fit(invalid)
    with pytest.raises(InsufficientResidualEvidence):
        HorizonResidualSampler(min_residuals_per_horizon=50).fit(_backtest())


def _proportional_backtest(n: int = 300) -> pd.DataFrame:
    # Errors proportional to the level: what count data with many zone sizes look like.
    rng = np.random.default_rng(3)
    level = np.exp(rng.uniform(0.0, 7.0, n))
    actual = rng.negative_binomial(5, 5 / (5 + level))
    return pd.DataFrame(
        {
            "actual": actual,
            "prediction": level,
            "horizon": 1,
            "forecast_origin": pd.Timestamp("2025-01-01"),
            "is_out_of_sample": True,
        }
    )


def test_log_scale_intervals_keep_their_coverage_for_large_counts() -> None:
    # New observations at 2,000 expected cases from the same overdispersed process:
    # an 80% interval should contain about 80% of them.
    rng = np.random.default_rng(11)
    future = rng.negative_binomial(5, 5 / (5 + 2000.0), size=4000)
    points = pd.DataFrame({"horizon": [1], "prediction": [2000.0]})

    def coverage(scale: str) -> float:
        sampler = HorizonResidualSampler(n_samples=1000, scale=scale).fit(_proportional_backtest())
        draws = sampler.add_samples(points)[[f"sample_{i}" for i in range(1000)]].to_numpy()[0]
        assert (draws >= 0).all()
        low, high = np.quantile(draws, [0.1, 0.9])
        return float(np.mean((future >= low) & (future <= high)))

    assert 0.7 <= coverage("log1p") <= 0.9
    # One pooled absolute distribution, dominated by small zones, is far too narrow here.
    assert coverage("additive") < 0.4


def test_level_bins_fall_back_to_one_pool_when_evidence_is_thin() -> None:
    rich = HorizonResidualSampler(level_bins=3, min_residuals_per_horizon=20)
    rich.fit(_proportional_backtest(300))
    thin = HorizonResidualSampler(level_bins=3, min_residuals_per_horizon=20)
    thin.fit(_proportional_backtest(40))
    assert len(rich.residuals_[1]) == 3
    assert len(thin.residuals_[1]) == 2


def test_sampler_round_trips_through_its_descriptor() -> None:
    fitted = HorizonResidualSampler(n_samples=30, seed=4).fit(_proportional_backtest())
    restored = HorizonResidualSampler.from_dict(fitted.to_dict())
    points = pd.DataFrame({"horizon": [1, 1, 1], "prediction": [0.0, 12.0, 900.0]})
    columns = [f"sample_{i}" for i in range(30)]
    assert np.array_equal(
        fitted.add_samples(points)[columns].to_numpy(),
        restored.add_samples(points)[columns].to_numpy(),
    )
    tampered = fitted.to_dict()
    tampered["method"] = "something_else"
    with pytest.raises(ValueError, match="unsupported"):
        HorizonResidualSampler.from_dict(tampered)
