"""Small reproducible benchmark CLI for the required synthetic scales."""

from __future__ import annotations

import argparse
import asyncio
import json
import pickle
import resource
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import replace

import pandas as pd

from chap_disease_forecasting.data import add_iso_weeks
from chap_disease_forecasting.features import WeeklyFeatureBuilder
from chap_disease_forecasting.forecasting import DirectForecaster
from chap_disease_forecasting.models.base import NativeRegressor
from chap_disease_forecasting.models.lightgbm import LightGBMConfig, LightGBMRegressor
from chap_disease_forecasting.models.xgboost import XGBoostConfig, XGBoostRegressor
from chap_disease_forecasting.synthetic import (
    generate_synthetic_history,
    preset_config,
    synthetic_geojson,
)


def _max_rss_mib() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    # Darwin reports bytes; Linux reports KiB.
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def _future_frame(history: pd.DataFrame, horizons: int = 3) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for location, group in history.groupby("location", observed=True):
        last = group.sort_values("time_period").iloc[-1]
        for horizon in range(1, horizons + 1):
            rows.append(
                {
                    "location": str(location),
                    "time_period": add_iso_weeks(last["time_period"], horizon),
                    "rainfall": float(group["rainfall"].mean()),
                    "mean_temperature": float(group["mean_temperature"].mean()),
                    "population": float(last["population"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def _run_through_service(
    history: pd.DataFrame,
    engine: str,
    n_estimators: int,
    n_threads: int,
    periods: int,
    geo: dict[str, object] | None = None,
) -> dict[str, object]:
    """Time the CHAPKit callbacks exactly as a CHAP train then predict job runs them."""

    from chapkit.data import DataFrame

    from chap_disease_forecasting.chapkit_adapter.service import (
        DiseaseModelConfig,
        on_predict,
        on_train,
    )

    config = DiseaseModelConfig.model_validate(
        {
            "engine": engine,
            "n_estimators": n_estimators,
            "n_threads": n_threads,
            "prediction_periods": periods,
        }
    )
    started = time.perf_counter()
    descriptor = asyncio.run(on_train(config, DataFrame.from_pandas(history), geo))  # type: ignore[arg-type]
    train_seconds = time.perf_counter() - started
    started = time.perf_counter()
    output = asyncio.run(
        on_predict(
            config,
            descriptor,
            DataFrame.from_pandas(history),
            DataFrame.from_pandas(_future_frame(history, periods)),
        )
    ).to_pandas()
    return {
        "service_train_seconds": train_seconds,
        "service_predict_seconds": time.perf_counter() - started,
        "predictions": len(output),
        "samples": sum(str(column).startswith("sample_") for column in output.columns),
        "locations_with_neighbours": descriptor["spatial"]["locations_with_neighbours"],
        "selected_models": descriptor["selection"]["selected_models"],
        "training_errors": descriptor["training_errors"],
        "descriptor_pickle_mib": len(pickle.dumps(descriptor)) / (1024.0 * 1024.0),
    }


def run_benchmark(
    zones: int,
    engine: str = "features",
    n_estimators: int = 20,
    seed: int = 20261025,
    through_service: bool = False,
    n_threads: int = 1,
    periods: int = 3,
) -> dict[str, object]:
    """Measure wall time and process maximum RSS for one required scale."""

    config = replace(preset_config(zones, seed=seed), missing_rate=0.005)
    rss_before = _max_rss_mib()
    started = time.perf_counter()
    history = generate_synthetic_history(config)
    generation_seconds = time.perf_counter() - started
    feature_started = time.perf_counter()
    features = WeeklyFeatureBuilder().fit_transform(history)
    feature_seconds = time.perf_counter() - feature_started
    training_seconds: float | None = None
    predictions = 0
    service: dict[str, object] = {}
    if through_service:
        if engine == "features":
            raise ValueError("--through-service needs xgboost, lightgbm or auto")
        service = _run_through_service(
            history, engine, n_estimators, n_threads, periods, synthetic_geojson(config)
        )
    elif engine == "auto":
        raise ValueError("engine auto is only available with --through-service")
    elif engine != "features":
        if engine == "xgboost":
            factory: Callable[[], NativeRegressor] = lambda: XGBoostRegressor(  # noqa: E731
                XGBoostConfig(n_estimators=n_estimators, n_jobs=1)
            )
        elif engine == "lightgbm":
            factory = lambda: LightGBMRegressor(  # noqa: E731
                LightGBMConfig(n_estimators=n_estimators, n_jobs=1)
            )
        else:
            raise ValueError("engine must be features, xgboost, or lightgbm")
        training_started = time.perf_counter()
        model = DirectForecaster(factory).fit(history)
        predictions = len(model.predict(history))
        training_seconds = time.perf_counter() - training_started
    return {
        "zones": zones,
        "weeks": config.n_weeks,
        "rows": len(history),
        "features": features.shape[1],
        "engine": engine,
        "n_estimators": n_estimators if engine != "features" else None,
        "generation_seconds": generation_seconds,
        "feature_seconds": feature_seconds,
        "training_prediction_seconds": training_seconds,
        "predictions": predictions,
        "through_service": through_service,
        "n_threads": n_threads if through_service else 1,
        "periods": periods if through_service else 3,
        **service,
        "max_rss_mib": _max_rss_mib(),
        "max_rss_increase_mib": max(0.0, _max_rss_mib() - rss_before),
        "seed": seed,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zones", type=int, choices=(10, 100, 514), default=10)
    parser.add_argument(
        "--engine", choices=("features", "xgboost", "lightgbm", "auto"), default="features"
    )
    parser.add_argument("--n-estimators", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument(
        "--through-service",
        action="store_true",
        help="run the CHAPKit on_train/on_predict callbacks, including selection backtests",
    )
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--periods", type=int, default=3, help="weeks ahead, with --through-service (1 to 12)"
    )
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run_benchmark(
                args.zones,
                args.engine,
                args.n_estimators,
                args.seed,
                args.through_service,
                args.threads,
                args.periods,
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
