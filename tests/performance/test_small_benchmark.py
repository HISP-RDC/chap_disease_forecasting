from __future__ import annotations

import pytest

from chap_disease_forecasting.evaluation.benchmark import run_benchmark


@pytest.mark.performance
def test_ten_zone_feature_benchmark_stays_bounded() -> None:
    result = run_benchmark(10, engine="features")
    assert result["rows"] == 10 * 52
    assert result["features"] >= 20
    assert result["feature_seconds"] < 30
    assert result["max_rss_increase_mib"] < 512
