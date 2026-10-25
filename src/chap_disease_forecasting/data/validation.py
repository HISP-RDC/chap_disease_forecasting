"""Validation for the tabular weekly data consumed by the ML core."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .weeks import normalize_week

REQUIRED_COLUMNS = ("time_period", "location", "disease_cases")
NUMERIC_COLUMNS = (
    "disease_cases",
    "rainfall",
    "mean_temperature",
    "population",
    "completeness",
)


class DataValidationError(ValueError):
    """Raised when an input table violates the forecasting data contract."""

    def __init__(self, messages: Iterable[str]):
        self.messages = tuple(messages)
        super().__init__("; ".join(self.messages))


@dataclass(frozen=True)
class ValidationConfig:
    """Controls validation without silently modifying observations."""

    require_contiguous_weeks: bool = True
    allow_missing_cases: bool = True
    require_population: bool = False
    known_locations: frozenset[str] | None = None
    allow_unknown_locations: bool = True


def _missing_week_messages(frame: pd.DataFrame) -> list[str]:
    messages: list[str] = []
    for location, group in frame.groupby("location", sort=False, observed=True):
        periods = pd.to_datetime(group["time_period"]).sort_values()
        if len(periods) < 2:
            continue
        deltas = periods.diff().dropna().dt.days
        invalid = deltas[deltas != 7]
        if not invalid.empty:
            messages.append(
                f"location {location!r} has {len(invalid)} non-contiguous weekly interval(s)"
            )
    return messages


def validate_history(
    data: pd.DataFrame,
    config: ValidationConfig | None = None,
) -> pd.DataFrame:
    """Validate and canonicalize weekly history.

    Missing observations remain missing; in particular, missing disease cases
    are never converted to zero. A defensive copy sorted by location and week
    is returned.
    """

    config = config or ValidationConfig()
    if not isinstance(data, pd.DataFrame):
        raise TypeError("data must be a pandas DataFrame")
    missing_columns = [name for name in REQUIRED_COLUMNS if name not in data.columns]
    if missing_columns:
        raise DataValidationError([f"missing required column(s): {', '.join(missing_columns)}"])

    frame = data.copy(deep=True)
    messages: list[str] = []
    if frame.empty:
        raise DataValidationError(["history is empty"])

    location_missing = frame["location"].isna() | frame["location"].astype(str).str.strip().eq("")
    if location_missing.any():
        messages.append(f"location is missing in {int(location_missing.sum())} row(s)")
    frame["location"] = frame["location"].astype("string")

    normalized_periods: list[Any] = []
    invalid_periods = 0
    for value in frame["time_period"]:
        try:
            normalized_periods.append(normalize_week(value))
        except ValueError:
            normalized_periods.append(pd.NaT)
            invalid_periods += 1
    frame["time_period"] = pd.to_datetime(normalized_periods)
    if invalid_periods:
        messages.append(f"time_period is invalid or missing in {invalid_periods} row(s)")

    for column in NUMERIC_COLUMNS:
        if column not in frame.columns:
            continue
        original_not_missing = frame[column].notna()
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        newly_missing = original_not_missing & frame[column].isna()
        if newly_missing.any():
            messages.append(f"{column} is non-numeric in {int(newly_missing.sum())} row(s)")
        finite = frame[column].dropna().map(np.isfinite)
        if not finite.all():
            messages.append(f"{column} contains non-finite value(s)")

    if not config.allow_missing_cases and frame["disease_cases"].isna().any():
        messages.append("disease_cases contains missing value(s)")
    if (frame["disease_cases"].dropna() < 0).any():
        messages.append("disease_cases contains negative value(s)")
    if "rainfall" in frame and (frame["rainfall"].dropna() < 0).any():
        messages.append("rainfall contains negative value(s)")
    if "population" in frame:
        if (frame["population"].dropna() <= 0).any():
            messages.append("population must be strictly positive when provided")
    elif config.require_population:
        messages.append("population is required")
    if "completeness" in frame:
        valid_completeness = frame["completeness"].dropna().between(0.0, 1.0)
        if not valid_completeness.all():
            messages.append("completeness must be between 0 and 1")

    if frame.duplicated(["location", "time_period"], keep=False).any():
        messages.append("duplicate (location, time_period) rows")

    if config.known_locations is not None:
        observed = set(frame["location"].dropna().astype(str))
        unknown = sorted(observed.difference(config.known_locations))
        if unknown and not config.allow_unknown_locations:
            preview = ", ".join(unknown[:5])
            messages.append(f"unknown location identifier(s): {preview}")

    if not invalid_periods and config.require_contiguous_weeks:
        messages.extend(_missing_week_messages(frame))
    if messages:
        raise DataValidationError(messages)

    return frame.sort_values(["location", "time_period"], kind="stable").reset_index(drop=True)
