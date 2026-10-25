"""Transparent point-forecast baselines and final fallback models."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from chap_disease_forecasting.data import add_iso_weeks, validate_history


def _forecast_rows(
    history: pd.DataFrame, horizons: tuple[int, ...], seasonal_lag: int | None
) -> pd.DataFrame:
    frame = validate_history(history)
    rows: list[dict[str, object]] = []
    for location, group in frame.groupby("location", sort=False, observed=True):
        group = group.sort_values("time_period")
        observed = group["disease_cases"].dropna()
        if observed.empty:
            raise ValueError(f"no observed disease_cases for {location!r}")
        latest_period = group["time_period"].max()
        for horizon in horizons:
            if seasonal_lag is None:
                prediction = float(observed.iloc[-1])
            else:
                target_period = add_iso_weeks(latest_period, horizon)
                reference_period = add_iso_weeks(target_period, -seasonal_lag)
                reference = group.loc[
                    group["time_period"].eq(reference_period), "disease_cases"
                ].dropna()
                prediction = float(reference.iloc[-1] if not reference.empty else observed.iloc[-1])
            rows.append(
                {
                    "location": str(location),
                    "time_period": add_iso_weeks(latest_period, horizon),
                    "horizon": horizon,
                    "prediction": max(prediction, 0.0),
                }
            )
    return pd.DataFrame.from_records(rows)


@dataclass
class PersistenceForecaster:
    horizons: tuple[int, ...] = (1, 2, 3)
    model_name: str = "naive_persistence"

    def fit(self, history: pd.DataFrame) -> PersistenceForecaster:
        validate_history(history)
        return self

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        output = _forecast_rows(history, self.horizons, seasonal_lag=None)
        output["model_name"] = self.model_name
        output["used_fallback"] = False
        return output


@dataclass
class SeasonalNaiveForecaster:
    horizons: tuple[int, ...] = (1, 2, 3)
    season_length: int = 52
    model_name: str = "naive_seasonal"

    def fit(self, history: pd.DataFrame) -> SeasonalNaiveForecaster:
        validate_history(history)
        return self

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        output = _forecast_rows(history, self.horizons, seasonal_lag=self.season_length)
        output["model_name"] = self.model_name
        output["used_fallback"] = False
        return output
