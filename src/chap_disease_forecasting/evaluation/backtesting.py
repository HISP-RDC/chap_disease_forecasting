"""Expanding-window rolling-origin evaluation."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from chap_disease_forecasting.data import validate_history

from .metrics import point_metrics


class TrainableForecaster(Protocol):
    def fit(self, history: pd.DataFrame) -> TrainableForecaster: ...

    def predict(self, history: pd.DataFrame) -> pd.DataFrame: ...


@dataclass(frozen=True)
class BacktestConfig:
    n_splits: int = 7
    horizon: int = 3
    min_train_weeks: int = 52
    step_weeks: int = 1

    def __post_init__(self) -> None:
        if min(self.n_splits, self.horizon, self.min_train_weeks, self.step_weeks) <= 0:
            raise ValueError("backtest configuration values must be positive")


class RollingOriginBacktester:
    def __init__(self, config: BacktestConfig | None = None):
        self.config = config or BacktestConfig()

    def split_origins(self, history: pd.DataFrame) -> tuple[pd.Timestamp, ...]:
        frame = validate_history(history)
        periods = list(pd.Index(frame["time_period"].unique()).sort_values())
        latest_origin_index = len(periods) - self.config.horizon - 1
        earliest_origin_index = self.config.min_train_weeks - 1
        indices = [
            latest_origin_index - offset * self.config.step_weeks
            for offset in reversed(range(self.config.n_splits))
        ]
        if not indices or min(indices) < earliest_origin_index:
            raise ValueError("history is too short for the requested rolling-origin splits")
        return tuple(pd.Timestamp(periods[index]) for index in indices)

    def run(
        self,
        history: pd.DataFrame,
        forecaster_factory: Callable[[], TrainableForecaster],
    ) -> pd.DataFrame:
        frame = validate_history(history)
        records: list[pd.DataFrame] = []
        for split_index, origin in enumerate(self.split_origins(frame)):
            train = frame.loc[frame["time_period"] <= origin].copy()
            forecaster = forecaster_factory().fit(train)
            predicted = forecaster.predict(train)
            predicted = predicted.loc[predicted["horizon"] <= self.config.horizon].copy()
            truth = frame[["location", "time_period", "disease_cases"]].rename(
                columns={"disease_cases": "actual"}
            )
            result = predicted.merge(truth, on=["location", "time_period"], how="left")
            result["split"] = split_index
            result["forecast_origin"] = origin
            result["is_out_of_sample"] = True
            records.append(result)
        return pd.concat(records, ignore_index=True) if records else pd.DataFrame()

    @staticmethod
    def summarize(results: pd.DataFrame) -> pd.DataFrame:
        required = {"actual", "prediction", "horizon"}
        if not required.issubset(results.columns):
            raise ValueError(
                f"results missing columns: {sorted(required.difference(results.columns))}"
            )
        rows: list[dict[str, float | int]] = []
        for horizon, group in results.groupby("horizon", sort=True):
            metrics = point_metrics(group["actual"], group["prediction"])
            rows.append({"horizon": int(str(horizon)), **metrics})
        return pd.DataFrame.from_records(rows)
