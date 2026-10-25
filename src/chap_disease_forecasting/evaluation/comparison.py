"""Out-of-sample comparison of model variants, as the service trains and predicts.

For every outer rolling origin, each variant is trained on the data up to that
origin only: an inner rolling-origin backtest provides the residuals for the
predictive samples, the forecaster is fitted, and the following ``horizon``
weeks are scored against the observed cases. Results are synthetic-data
evidence for software choices, not validation on real surveillance data.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd

from chap_disease_forecasting.data import validate_history
from chap_disease_forecasting.features import FeatureConfig
from chap_disease_forecasting.features.spatial import neighbour_graph
from chap_disease_forecasting.forecasting import (
    DirectForecastConfig,
    DirectForecaster,
    HorizonResidualSampler,
)
from chap_disease_forecasting.models.base import NativeRegressor
from chap_disease_forecasting.models.lightgbm import LightGBMConfig, LightGBMRegressor
from chap_disease_forecasting.models.xgboost import XGBoostConfig, XGBoostRegressor
from chap_disease_forecasting.synthetic import (
    SyntheticConfig,
    generate_synthetic_history,
    synthetic_geojson,
)

from .backtesting import BacktestConfig, RollingOriginBacktester
from .metrics import empirical_crps, interval_coverage, interval_score

# The feature set shipped up to version 0.3.x.
LEGACY_FEATURES = FeatureConfig(
    climate_lags=tuple(range(1, 13)),
    climate_windows=(4, 8, 12),
    dynamics=False,
    spatial=False,
)


@dataclass(frozen=True)
class Variant:
    name: str
    features: FeatureConfig = field(default_factory=FeatureConfig)
    objective: Literal["poisson", "tweedie"] = "poisson"
    scale: Literal["additive", "log1p"] = "log1p"
    spatial: bool = True
    engine: Literal["xgboost", "lightgbm"] = "xgboost"


DEFAULT_VARIANTS = (
    Variant("v0.3 baseline", LEGACY_FEATURES, scale="additive", spatial=False),
    Variant("+ log-scale uncertainty", LEGACY_FEATURES, spatial=False),
    Variant("+ dynamics & climate", FeatureConfig(spatial=False), spatial=False),
    Variant("+ neighbours", FeatureConfig()),
    Variant("+ tweedie", FeatureConfig(), objective="tweedie"),
)


def _engine_factory(
    variant: Variant, n_estimators: int, seed: int
) -> Callable[[], NativeRegressor]:
    tweedie = variant.objective == "tweedie"
    if variant.engine == "xgboost":
        xgb_config = XGBoostConfig(
            n_estimators=n_estimators,
            random_state=seed,
            objective="reg:tweedie" if tweedie else "count:poisson",
            tweedie_variance_power=1.5 if tweedie else None,
        )
        return lambda: XGBoostRegressor(xgb_config)
    lgb_config = LightGBMConfig(
        n_estimators=n_estimators,
        random_state=seed,
        objective=variant.objective,
        tweedie_variance_power=1.5 if tweedie else None,
    )
    return lambda: LightGBMRegressor(lgb_config)


def evaluate_variant(
    history: pd.DataFrame,
    variant: Variant,
    *,
    neighbours: dict[str, list[str]] | None = None,
    outer_splits: int = 4,
    inner_splits: int = 3,
    horizon: int = 3,
    inner_min_train_weeks: int = 104,
    n_estimators: int = 120,
    n_samples: int = 200,
    min_residuals_per_horizon: int = 20,
    seed: int = 20261025,
) -> dict[str, Any]:
    frame = validate_history(history)
    graph = neighbours if variant.spatial else None
    factory = _engine_factory(variant, n_estimators, seed)
    forecast_config = DirectForecastConfig(horizons=tuple(range(1, horizon + 1)))

    def make() -> DirectForecaster:
        return DirectForecaster(factory, forecast_config, variant.features, graph)

    outer_min_train = inner_min_train_weeks + inner_splits + horizon
    origins = RollingOriginBacktester(
        BacktestConfig(n_splits=outer_splits, horizon=horizon, min_train_weeks=outer_min_train)
    ).split_origins(frame)
    truth = frame[["location", "time_period", "disease_cases"]].rename(
        columns={"disease_cases": "actual"}
    )
    scored: list[pd.DataFrame] = []
    started = time.perf_counter()
    for origin in origins:
        train = frame.loc[frame["time_period"] <= origin]
        inner = RollingOriginBacktester(
            BacktestConfig(
                n_splits=inner_splits, horizon=horizon, min_train_weeks=inner_min_train_weeks
            )
        ).run(train, make)
        sampler = HorizonResidualSampler(
            n_samples=n_samples,
            min_residuals_per_horizon=min_residuals_per_horizon,
            seed=seed,
            scale=variant.scale,
        )
        sampler.fit(inner)
        forecast = sampler.add_samples(make().fit(train).predict(train))
        scored.append(forecast.merge(truth, on=["location", "time_period"], how="inner"))
    results = pd.concat(scored, ignore_index=True).dropna(subset=["actual"])
    sample_columns = [f"sample_{index}" for index in range(n_samples)]
    samples = results[sample_columns].to_numpy(dtype=float)
    actual = results["actual"].to_numpy(dtype=float)
    point = results["prediction"].to_numpy(dtype=float)
    q = {level: np.quantile(samples, level, axis=1) for level in (0.1, 0.25, 0.75, 0.9)}
    by_horizon = {
        int(str(h)): round(float(np.mean(np.abs(group["prediction"] - group["actual"]))), 4)
        for h, group in results.groupby("horizon")
    }
    return {
        "variant": variant.name,
        "n_forecasts": len(results),
        "mae": round(float(np.mean(np.abs(point - actual))), 4),
        "rmse": round(float(np.sqrt(np.mean((point - actual) ** 2))), 4),
        "crps": round(empirical_crps(actual, samples), 4),
        "coverage_10_90": round(interval_coverage(actual, q[0.1], q[0.9]), 4),
        "coverage_25_75": round(interval_coverage(actual, q[0.25], q[0.75]), 4),
        "interval_score_10_90": round(interval_score(actual, q[0.1], q[0.9], alpha=0.2), 4),
        "mae_by_horizon": by_horizon,
        "seconds": round(time.perf_counter() - started, 1),
    }


def compare(
    disease: Literal["malaria", "cholera"],
    zones: int = 100,
    weeks: int = 196,
    seed: int = 20261025,
    variants: Sequence[Variant] = DEFAULT_VARIANTS,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    config = SyntheticConfig(
        n_locations=zones,
        n_weeks=weeks,
        seed=seed,
        disease=disease,
        short_history_fraction=0.0,
        missing_rate=0.005,
    )
    history = generate_synthetic_history(config)
    graph = neighbour_graph(
        synthetic_geojson(config), 5, sorted(history["location"].astype(str).unique())
    )
    rows = []
    for variant in variants:
        result = evaluate_variant(history, variant, neighbours=graph, seed=seed, **kwargs)
        rows.append({"disease": disease, **result})
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--disease", choices=("malaria", "cholera"), default="malaria")
    parser.add_argument("--zones", type=int, default=100)
    parser.add_argument("--weeks", type=int, default=196)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--outer-splits", type=int, default=4)
    parser.add_argument("--n-estimators", type=int, default=120)
    args = parser.parse_args(argv)
    for row in compare(
        args.disease,
        args.zones,
        args.weeks,
        args.seed,
        outer_splits=args.outer_splits,
        n_estimators=args.n_estimators,
    ):
        print(json.dumps(row, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
