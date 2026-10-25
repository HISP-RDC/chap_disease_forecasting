"""ISO-week handling with Monday as the canonical timestamp."""

from __future__ import annotations

import datetime as dt
import re
from typing import Any

import pandas as pd

_ISO_WEEK_RE = re.compile(r"^(?P<year>\d{4})-?W(?P<week>\d{1,2})(?:-?[1-7])?$")


def normalize_week(value: Any) -> pd.Timestamp:
    """Return the Monday of *value*'s ISO week.

    Strings such as ``2024-W01`` are parsed as ISO weeks. Date-like values are
    normalized to the Monday of their containing ISO week. Timezones are
    removed because CHAP periods are calendar labels rather than instants.
    """

    if value is None or pd.isna(value):
        raise ValueError("time_period cannot be missing")
    if isinstance(value, str):
        match = _ISO_WEEK_RE.fullmatch(value.strip())
        if match:
            year = int(match.group("year"))
            week = int(match.group("week"))
            try:
                return pd.Timestamp(dt.date.fromisocalendar(year, week, 1))
            except ValueError as exc:
                raise ValueError(f"invalid ISO week: {value!r}") from exc
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid weekly period: {value!r}") from exc
    if pd.isna(timestamp):
        raise ValueError("time_period cannot be missing")
    if timestamp.tz is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize() - pd.Timedelta(days=timestamp.weekday())


def iso_week_label(value: Any) -> str:
    """Format a date-like value as an unambiguous ``YYYY-Www`` label."""

    iso = normalize_week(value).isocalendar()
    return f"{int(iso.year):04d}-W{int(iso.week):02d}"


def add_iso_weeks(value: Any, weeks: int) -> pd.Timestamp:
    """Add whole calendar weeks to a normalized weekly period."""

    if not isinstance(weeks, int):
        raise TypeError("weeks must be an integer")
    return normalize_week(value) + pd.Timedelta(weeks=weeks)
