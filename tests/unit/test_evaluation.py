from __future__ import annotations

import numpy as np

from chap_disease_forecasting.evaluation import (
    BacktestConfig,
    RollingOriginBacktester,
    alert_metrics,
    empirical_crps,
    interval_coverage,
    interval_score,
    point_metrics,
)
from chap_disease_forecasting.models import PersistenceForecaster
from chap_disease_forecasting.synthetic import SyntheticConfig, generate_synthetic_history


def test_point_and_probabilistic_metrics() -> None:
    metrics = point_metrics([0, 10, 20], [2, 8, 25])
    assert metrics["n"] == 3
    assert metrics["mape_n"] == 2
    assert metrics["mae"] == 3.0
    samples = np.array([[0, 1, 2], [8, 10, 12], [18, 20, 22]], dtype=float)
    assert empirical_crps([1, 10, 20], samples) >= 0
    assert interval_coverage([1, 10], [0, 8], [2, 12]) == 1.0
    assert interval_score([1, 10], [0, 8], [2, 12], alpha=0.1) == 3.0
    alerts = alert_metrics([5, 15], [12, 20], [10, 10])
    assert alerts["sensitivity"] == 1.0
    assert alerts["false_positive_rate"] == 1.0


def test_rolling_origin_backtest_uses_only_past_training_data() -> None:
    history = generate_synthetic_history(
        SyntheticConfig(
            n_locations=2,
            n_weeks=60,
            missing_rate=0,
            short_history_fraction=0,
            seed=17,
        )
    )
    backtester = RollingOriginBacktester(BacktestConfig(n_splits=2, horizon=3, min_train_weeks=40))
    results = backtester.run(history, PersistenceForecaster)
    assert set(results["horizon"]) == {1, 2, 3}
    assert results["is_out_of_sample"].all()
    assert (results["time_period"] > results["forecast_origin"]).all()
    summary = backtester.summarize(results)
    assert set(summary["horizon"]) == {1, 2, 3}
