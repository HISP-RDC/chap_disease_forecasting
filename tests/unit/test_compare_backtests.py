from __future__ import annotations

import importlib.util
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from chap_disease_forecasting.evaluation import empirical_crps

ROOT = Path(__file__).resolve().parents[2]


def _module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "compare_backtests", ROOT / "deploy" / "compare_backtests.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _backtest(spread: float, bias: float, truth: dict[tuple[str, str], float]) -> dict[str, Any]:
    rng = random.Random(int(spread * 10 + bias))  # noqa: S311
    forecasts = []
    for (zone, period), value in truth.items():
        centre = value + bias
        forecasts.append(
            {
                "orgUnit": zone,
                "period": period,
                "lastSeenPeriod": "2026W30",
                "values": [max(0.0, rng.gauss(centre, spread)) for _ in range(100)],
            }
        )
    return {"modelId": f"model-{spread}", "forecasts": forecasts}


def _truth() -> dict[tuple[str, str], float]:
    rng = random.Random(1)  # noqa: S311
    return {
        (f"Z{zone:02d}", f"2026W{week}"): float(rng.randint(0, 40))
        for zone in range(30)
        for week in (31, 32, 33)
    }


def test_crps_matches_the_library_and_periods_order_across_years() -> None:
    module = _module()
    samples = [0.0, 1.0, 4.0, 9.0, 2.5]
    assert module.crps(samples, 3.0) == pytest.approx(empirical_crps([3.0], np.array([samples])))
    # 2026 has 53 ISO weeks, 2025 has 52.
    assert module.week_index("2027W01") - module.week_index("2026W53") == 1
    assert module.week_index("2026W01") - module.week_index("2025W52") == 1
    assert module.week_index("2026-W05") == module.week_index("2026W5")


def test_sign_test_values() -> None:
    module = _module()
    assert module.sign_test(10, 10) == pytest.approx(1.0)
    assert module.sign_test(9, 1) == pytest.approx(0.021484375)
    assert module.sign_test(0, 0) == 1.0


def test_a_clearly_better_backtest_is_detected() -> None:
    module = _module()
    truth = _truth()
    observed = {"data": [{"ou": z, "pe": p, "value": v} for (z, p), v in truth.items()]}
    good = module.forecasts_by_key(_backtest(2.0, 0.0, truth))
    poor = module.forecasts_by_key(_backtest(15.0, 8.0, truth))
    result = module.compare(good, poor, module.observed_by_key(observed), resamples=300)
    low, high = result["crps_ratio_ci95"]
    assert result["matched_forecasts"] == 90 and result["zones"] == 30
    assert high < 1.0 and low > 0.0
    assert result["zones_won"][0] > 25 and result["sign_test_p"] < 0.001
    assert set(result["crps_by_horizon"]) == {1, 2, 3}
    assert "95% CI" in module.report(result, ("good", "poor"))


def test_equivalent_backtests_are_not_declared_different() -> None:
    module = _module()
    truth = _truth()
    observed = module.observed_by_key(
        {"data": [{"ou": z, "pe": p, "value": v} for (z, p), v in truth.items()]}
    )
    first = module.forecasts_by_key(_backtest(5.0, 0.0, truth))
    second = module.forecasts_by_key(_backtest(5.0, 0.0, truth) | {"modelId": "copy"})
    low, high = module.compare(first, second, observed, resamples=300)["crps_ratio_ci95"]
    assert low <= 1.0 <= high
