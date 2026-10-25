"""Feature generation that never looks beyond the row's forecast origin."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd

from chap_disease_forecasting.data import ValidationConfig, validate_history


@dataclass(frozen=True)
class FeatureConfig:
    case_lags: tuple[int, ...] = (1, 2, 3, 4, 8, 52)
    case_windows: tuple[int, ...] = (4, 8, 12)
    # Lag 0 is the origin week itself, observed when the forecast is made. Short
    # lags matter for cholera (flooding, contamination); the 12-16 week windows
    # cover malaria's vector and parasite development delays.
    climate_lags: tuple[int, ...] = tuple(range(0, 13))
    climate_windows: tuple[int, ...] = (4, 8, 12, 16)
    climate_columns: tuple[str, ...] = ("rainfall", "mean_temperature")
    include_cases_at_origin: bool = True
    # Growth rates, current incidence and climate anomalies.
    dynamics: bool = True
    # Neighbour features, used only when a neighbour graph is supplied.
    spatial: bool = True


class WeeklyFeatureBuilder:
    """Build numerical features independently within each health zone.

    Rolling summaries are shifted by one week. ``cases_at_origin`` is the one
    observation allowed at the forecast origin and is only used to predict
    strictly later weeks in the direct forecasting pipeline. Neighbour features
    use the neighbours' values up to the same origin week, never later ones.
    """

    def __init__(
        self,
        config: FeatureConfig | None = None,
        neighbours: dict[str, list[str]] | None = None,
    ):
        self.config = config or FeatureConfig()
        self.neighbours_: dict[str, list[str]] = dict(neighbours or {})
        self.location_mapping_: dict[str, int] | None = None
        self.parent_mapping_: dict[str, int] | None = None
        self.feature_names_: tuple[str, ...] | None = None
        self.time_origin_: pd.Timestamp | None = None

    @property
    def config_dict(self) -> dict[str, Any]:
        return asdict(self.config)

    def fit(self, history: pd.DataFrame) -> WeeklyFeatureBuilder:
        frame = validate_history(history, ValidationConfig(require_contiguous_weeks=False))
        locations = sorted(frame["location"].dropna().astype(str).unique())
        self.location_mapping_ = {value: index for index, value in enumerate(locations)}
        if "parent" in frame:
            parents = sorted(frame["parent"].dropna().astype(str).unique())
            self.parent_mapping_ = {value: index for index, value in enumerate(parents)}
        else:
            self.parent_mapping_ = {}
        self.time_origin_ = frame["time_period"].min()
        features = self._build(frame)
        self.feature_names_ = tuple(features.columns)
        return self

    def fit_transform(self, history: pd.DataFrame) -> pd.DataFrame:
        self.fit(history)
        return self.transform(history)

    def transform(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.location_mapping_ is None or self.parent_mapping_ is None:
            raise RuntimeError("WeeklyFeatureBuilder must be fitted before transform")
        frame = validate_history(history, ValidationConfig(require_contiguous_weeks=False))
        features = self._build(frame)
        if self.feature_names_ is None:
            self.feature_names_ = tuple(features.columns)
        missing = set(self.feature_names_).difference(features.columns)
        if missing:
            raise ValueError(f"feature schema changed; missing: {sorted(missing)}")
        return features.loc[:, self.feature_names_]

    def _build(self, frame: pd.DataFrame) -> pd.DataFrame:
        assert self.location_mapping_ is not None
        assert self.parent_mapping_ is not None
        assert self.time_origin_ is not None
        group = frame.groupby("location", sort=False, observed=True)
        cases = frame["disease_cases"].astype(float)
        result = pd.DataFrame(index=frame.index)

        if self.config.include_cases_at_origin:
            result["cases_at_origin"] = cases
            result["cases_at_origin_missing"] = cases.isna().astype(np.int8)
        for lag in self.config.case_lags:
            values = group["disease_cases"].shift(lag).astype(float)
            result[f"cases_lag_{lag}"] = values
            result[f"cases_lag_{lag}_missing"] = values.isna().astype(np.int8)

        shifted_cases = group["disease_cases"].shift(1).astype(float)
        shifted_group = shifted_cases.groupby(frame["location"], sort=False, observed=True)
        for window in self.config.case_windows:
            rolling = shifted_group.rolling(window=window, min_periods=1)
            result[f"cases_mean_{window}"] = rolling.mean().reset_index(level=0, drop=True)
            result[f"cases_std_{window}"] = (
                rolling.std(ddof=0).reset_index(level=0, drop=True).fillna(0.0)
            )

        lag_1 = result["cases_lag_1"]
        lag_4 = result["cases_lag_4"]
        result["cases_trend_1_4"] = lag_1 - lag_4
        denominator = lag_4.abs().where(lag_4.abs() > 0)
        result["cases_relative_change_1_4"] = (lag_1 - lag_4) / denominator

        if "population" in frame:
            population = frame["population"].astype(float)
            result["log_population"] = np.log(population)
            result["incidence_lag_1_per_1000"] = 1_000.0 * lag_1 / population
            result["population_missing"] = population.isna().astype(np.int8)

        if self.config.dynamics:
            log_cases = np.log1p(cases)
            mean_4 = (
                shifted_group.rolling(window=4, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )
            result["log_growth_1w"] = log_cases - np.log1p(lag_1)
            result["log_growth_4w"] = log_cases - np.log1p(mean_4)
            if "population" in frame:
                result["incidence_at_origin_per_1000"] = (
                    1_000.0 * cases / frame["population"].astype(float)
                )

        for column in self.config.climate_columns:
            if column not in frame:
                continue
            values = frame[column].astype(float)
            climate_group = values.groupby(frame["location"], sort=False, observed=True)
            for lag in self.config.climate_lags:
                lagged = climate_group.shift(lag)
                result[f"{column}_lag_{lag}"] = lagged
                result[f"{column}_lag_{lag}_missing"] = lagged.isna().astype(np.int8)
            shifted = climate_group.shift(1)
            shifted_climate_group = shifted.groupby(frame["location"], sort=False, observed=True)
            for window in self.config.climate_windows:
                result[f"{column}_mean_{window}"] = (
                    shifted_climate_group.rolling(window=window, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
            if self.config.dynamics:
                # Last four weeks up to the origin against the location's own
                # history before them: heavy rain or heat relative to normal.
                recent = (
                    climate_group.rolling(window=4, min_periods=1)
                    .mean()
                    .reset_index(level=0, drop=True)
                )
                expanding = shifted_climate_group.expanding(min_periods=8)
                normal = expanding.mean().reset_index(level=0, drop=True)
                spread = expanding.std(ddof=0).reset_index(level=0, drop=True)
                result[f"{column}_anomaly_4"] = (recent - normal) / spread.where(spread > 0)

        iso = frame["time_period"].dt.isocalendar()
        week = iso.week.astype(float)
        result["week_sin"] = np.sin(2.0 * np.pi * week / 52.1775)
        result["week_cos"] = np.cos(2.0 * np.pi * week / 52.1775)
        result["time_index"] = (frame["time_period"] - self.time_origin_).dt.days / 7.0
        result["location_code"] = (
            frame["location"].astype(str).map(self.location_mapping_).fillna(-1).astype(float)
        )
        result["location_unknown"] = (
            ~frame["location"].astype(str).isin(self.location_mapping_)
        ).astype(np.int8)
        if "parent" in frame:
            parents = frame["parent"].astype("string")
            result["parent_code"] = (
                parents.astype(str).map(self.parent_mapping_).fillna(-1).astype(float)
            )
            result["parent_missing"] = parents.isna().astype(np.int8)
            result["parent_unknown"] = (
                parents.notna() & ~parents.astype(str).isin(self.parent_mapping_)
            ).astype(np.int8)
        if "completeness" in frame:
            result["completeness_at_origin"] = frame["completeness"].astype(float)
            result["completeness_missing"] = frame["completeness"].isna().astype(np.int8)
        if self.config.spatial and self.neighbours_:
            result = result.join(self._neighbour_features(frame, cases))

        return result.astype(float)

    def _neighbour_features(self, frame: pd.DataFrame, cases: pd.Series) -> pd.DataFrame:
        """Mean incidence, recent level and growth of each location's neighbours."""

        if "population" in frame:
            level = 1_000.0 * cases / frame["population"].astype(float)
        else:
            level = cases
        keyed = pd.DataFrame(
            {
                "time_period": frame["time_period"],
                "location": frame["location"].astype(str),
                "level": level,
            }
        )
        wide = keyed.pivot(index="time_period", columns="location", values="level").sort_index()
        recent = wide.rolling(window=4, min_periods=1).mean()
        previous = wide.shift(1).rolling(window=4, min_periods=1).mean()
        growth = pd.DataFrame(
            np.log1p(wide.to_numpy(dtype=float)) - np.log1p(previous.to_numpy(dtype=float)),
            index=wide.index,
            columns=wide.columns,
        )

        columns = list(wide.columns)
        position = {name: index for index, name in enumerate(columns)}
        weights = np.zeros((len(columns), len(columns)), dtype=float)
        for name, neighbours in self.neighbours_.items():
            if name not in position:
                continue
            for neighbour in neighbours:
                if neighbour in position:
                    weights[position[name], position[neighbour]] = 1.0

        def neighbour_mean(values: pd.DataFrame) -> pd.DataFrame:
            matrix = values.to_numpy(dtype=float)
            observed = np.isfinite(matrix)
            total = np.where(observed, matrix, 0.0) @ weights.T
            count = observed.astype(float) @ weights.T
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(count > 0, total / count, np.nan)
            return pd.DataFrame(mean, index=values.index, columns=columns)

        aggregates = {
            "neighbours_incidence": neighbour_mean(wide),
            "neighbours_incidence_mean_4": neighbour_mean(recent),
            "neighbours_log_growth_4w": neighbour_mean(growth),
        }
        output = pd.DataFrame(index=frame.index)
        rows = pd.MultiIndex.from_arrays([keyed["time_period"], keyed["location"]])
        for name, table in aggregates.items():
            stacked = table.stack(future_stack=True)
            output[name] = stacked.reindex(rows).to_numpy()
        output["neighbours_pressure"] = np.log1p(output["neighbours_incidence_mean_4"]) - np.log1p(
            level.to_numpy()
        )
        output["neighbours_missing"] = output["neighbours_incidence"].isna().astype(np.int8)
        return output
