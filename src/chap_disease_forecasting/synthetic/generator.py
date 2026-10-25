"""Synthetic weekly disease data for software, scaling and comparison tests only.

The relationships below are chosen to exercise the pipeline (seasonality,
lagged climate, overdispersion, zeros, outbreaks, spatial spread). They are
not validated epidemiology and must not be used as evidence about real
malaria or cholera dynamics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SyntheticConfig:
    n_locations: int = 10
    n_weeks: int = 52
    start: str = "2022-01-03"
    seed: int = 20261025
    missing_rate: float = 0.01
    zero_inflation: float = 0.03
    short_history_fraction: float = 0.1
    overdispersion: float = 12.0
    # "malaria": seasonal, climate-driven, independent locations.
    # "cholera": endemic hotspots, many zeros, rain-triggered outbreaks that
    # spread to neighbouring locations on a grid.
    disease: Literal["malaria", "cholera"] = "malaria"

    def __post_init__(self) -> None:
        if self.n_locations <= 0 or self.n_weeks <= 3:
            raise ValueError("synthetic dataset dimensions must be positive")
        if not 0 <= self.missing_rate < 1 or not 0 <= self.zero_inflation < 1:
            raise ValueError("rates must be in [0, 1)")
        if self.overdispersion <= 0:
            raise ValueError("overdispersion must be positive")
        if self.disease not in {"malaria", "cholera"}:
            raise ValueError("disease must be 'malaria' or 'cholera'")


def _grid(n_locations: int) -> tuple[int, list[tuple[int, int]]]:
    side = math.ceil(math.sqrt(n_locations))
    return side, [(index // side, index % side) for index in range(n_locations)]


def _location_name(index: int) -> str:
    return f"ZS-{index + 1:04d}"


def synthetic_geojson(config: SyntheticConfig, cell_degrees: float = 0.5) -> dict[str, Any]:
    """Square polygons on a grid, one per synthetic location, ids = location names."""

    _, cells = _grid(config.n_locations)
    features = []
    for index, (row, column) in enumerate(cells):
        west, south = 18.0 + column * cell_degrees, -8.0 + row * cell_degrees
        ring = [
            [west, south],
            [west + cell_degrees, south],
            [west + cell_degrees, south + cell_degrees],
            [west, south + cell_degrees],
            [west, south],
        ]
        features.append(
            {
                "type": "Feature",
                "id": _location_name(index),
                "properties": {},
                "geometry": {"type": "Polygon", "coordinates": [ring]},
            }
        )
    return {"type": "FeatureCollection", "features": features}


def _grid_neighbours(n_locations: int) -> list[list[int]]:
    side, cells = _grid(n_locations)
    lookup = {cell: index for index, cell in enumerate(cells)}
    return [
        [
            lookup[(row + d_row, column + d_column)]
            for d_row, d_column in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if (row + d_row, column + d_column) in lookup
        ]
        for row, column in cells
    ]


def _generate_cholera(config: SyntheticConfig, rng: np.random.Generator) -> pd.DataFrame:
    n = config.n_locations
    periods = pd.date_range(config.start, periods=config.n_weeks, freq="W-MON")
    neighbours = _grid_neighbours(n)
    population = rng.integers(5_000, 600_000, size=n).astype(float)
    hotspot = rng.random(n) < 0.2
    baseline = np.where(hotspot, rng.lognormal(0.8, 0.6, n), 0.03)
    phase = rng.uniform(-np.pi, np.pi, n)
    region = np.array([index * 6 // max(n, 1) for index in range(n)])
    rain = np.zeros((config.n_weeks, n))
    temperature = np.zeros((config.n_weeks, n))
    cases = np.zeros((config.n_weeks, n))
    dispersion = 1.5
    for week in range(config.n_weeks):
        angle = 2.0 * np.pi * week / 52.1775 + phase
        storms = rng.random(6) < 0.06
        shock = np.where(storms[region], rng.uniform(60.0, 140.0, n), 0.0)
        rain[week] = np.maximum(0.0, 80.0 + 55.0 * np.sin(angle) + shock + rng.normal(0, 15, n))
        temperature[week] = 25.0 + 2.5 * np.cos(angle - 0.6) + rng.normal(0.0, 0.8, n)
        previous = cases[week - 1] if week else baseline
        neighbour_previous = np.array([previous[ids].mean() if ids else 0.0 for ids in neighbours])
        recent_rain = rain[max(0, week - 2) : week].mean(axis=0) if week else rain[0]
        flood = np.exp(np.clip(0.018 * (recent_rain - 95.0), -2.0, 2.5))
        # Rain-driven introductions, then self-limiting local and neighbour spread
        # (autoregressive weights sum to 0.65 < 1, so every outbreak dies down).
        seeding = baseline * flood * (1.0 + 0.3 * np.sin(angle))
        expected = np.maximum(0.01, seeding + 0.45 * previous + 0.2 * neighbour_previous)
        cases[week] = rng.negative_binomial(dispersion, dispersion / (dispersion + expected))
    rows = [
        {
            "time_period": periods[week],
            "location": _location_name(index),
            "disease_cases": float(cases[week, index]),
            "rainfall": float(rain[week, index]),
            "mean_temperature": float(temperature[week, index]),
            "population": float(population[index]),
        }
        for index in range(n)
        for week in range(config.n_weeks)
    ]
    return pd.DataFrame.from_records(rows)


def preset_config(level: int, *, seed: int = 20261025) -> SyntheticConfig:
    """Return required 10/100/514-zone configurations."""

    presets = {10: 52, 100: 196, 514: 196, 1000: 196}
    if level not in presets:
        raise ValueError(f"unsupported synthetic level: {level}")
    return SyntheticConfig(n_locations=level, n_weeks=presets[level], seed=seed)


def generate_synthetic_history(config: SyntheticConfig | None = None) -> pd.DataFrame:
    """Generate reproducible correlated-but-non-scientific weekly fixtures."""

    config = config or SyntheticConfig()
    rng = np.random.default_rng(config.seed)
    if config.disease == "cholera":
        frame = _generate_cholera(config, rng)
        if config.missing_rate:
            for column in ("disease_cases", "rainfall", "mean_temperature"):
                mask = rng.random(len(frame)) < config.missing_rate
                frame.loc[mask, column] = np.nan
        return frame.sort_values(["location", "time_period"], kind="stable", ignore_index=True)
    all_periods = pd.date_range(config.start, periods=config.n_weeks, freq="W-MON")
    rows: list[dict[str, object]] = []
    short_count = int(round(config.n_locations * config.short_history_fraction))
    for zone_index in range(config.n_locations):
        location = _location_name(zone_index)
        province = f"P-{zone_index % max(1, min(26, config.n_locations // 4 + 1)) + 1:02d}"
        population = float(rng.integers(1_500, 450_000))
        if zone_index % 25 == 0:
            population = float(rng.integers(250, 1_500))
        intensity = float(rng.lognormal(mean=2.2, sigma=0.65))
        phase = float(rng.uniform(-np.pi, np.pi))
        trend = float(rng.normal(0.0, 0.0015))
        short_start = 0
        if zone_index < short_count:
            short_start = min(config.n_weeks // 3, max(1, config.n_weeks - 8))
        rain_history: list[float] = []
        temp_history: list[float] = []
        cases_history: list[float] = []
        outbreak_start = int(rng.integers(max(4, config.n_weeks // 3), max(5, config.n_weeks - 5)))
        for absolute_week, period in enumerate(all_periods):
            angle = 2.0 * np.pi * absolute_week / 52.1775 + phase
            rainfall = max(0.0, 85.0 + 65.0 * np.sin(angle) + rng.normal(0.0, 18.0))
            temperature = 24.0 + 3.0 * np.cos(angle - 0.8) + rng.normal(0.0, 0.9)
            rain_history.append(rainfall)
            temp_history.append(temperature)
            lagged_rain = rain_history[-3] if len(rain_history) >= 3 else rainfall
            lagged_temp = temp_history[-2] if len(temp_history) >= 2 else temperature
            previous = cases_history[-1] if cases_history else intensity
            seasonal = 1.0 + 0.30 * np.sin(angle - 0.4)
            climate = np.exp(0.0025 * (lagged_rain - 85.0) + 0.025 * (lagged_temp - 24.0))
            outbreak = 1.8 if outbreak_start <= absolute_week < outbreak_start + 3 else 1.0
            expected = max(
                0.05,
                (0.65 * intensity * seasonal * climate + 0.35 * previous)
                * (1 + trend * absolute_week)
                * outbreak,
            )
            dispersion = config.overdispersion
            probability = dispersion / (dispersion + expected)
            cases = float(rng.negative_binomial(dispersion, probability))
            if rng.random() < config.zero_inflation:
                cases = 0.0
            cases_history.append(cases)
            is_pre_history = absolute_week < short_start
            rows.append(
                {
                    "time_period": period,
                    "location": location,
                    "parent": province,
                    "disease_cases": np.nan if is_pre_history else cases,
                    "rainfall": np.nan if is_pre_history else rainfall,
                    "mean_temperature": np.nan if is_pre_history else temperature,
                    "population": population,
                    "completeness": float(rng.uniform(0.82, 1.0)),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    if config.missing_rate:
        for column in ("disease_cases", "rainfall", "mean_temperature"):
            mask = rng.random(len(frame)) < config.missing_rate
            frame.loc[mask, column] = np.nan
    return frame.sort_values(["location", "time_period"], kind="stable", ignore_index=True)
