from __future__ import annotations

import pandas as pd
import pytest

from chap_disease_forecasting.data import (
    DataValidationError,
    ValidationConfig,
    add_iso_weeks,
    normalize_week,
    validate_history,
)


def _history() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "time_period": ["2020-W53", "2021-W01"],
            "location": ["A", "A"],
            "disease_cases": [0.0, None],
            "rainfall": [10.0, 5.0],
            "mean_temperature": [22.0, 23.0],
            "population": [1000.0, 1000.0],
        }
    )


def test_iso_week_crosses_year_correctly() -> None:
    assert normalize_week("2020-W53") == pd.Timestamp("2020-12-28")
    assert add_iso_weeks("2020-W53", 1) == pd.Timestamp("2021-01-04")
    assert normalize_week("2021-W01") == pd.Timestamp("2021-01-04")


def test_validation_preserves_zero_and_missing_cases() -> None:
    result = validate_history(_history())
    assert result.loc[0, "disease_cases"] == 0.0
    assert pd.isna(result.loc[1, "disease_cases"])


@pytest.mark.parametrize(
    ("column", "value", "message"),
    [
        ("disease_cases", -1, "negative"),
        ("rainfall", -0.1, "negative"),
        ("population", 0, "strictly positive"),
    ],
)
def test_validation_rejects_impossible_values(column: str, value: float, message: str) -> None:
    frame = _history()
    frame.loc[0, column] = value
    with pytest.raises(DataValidationError, match=message):
        validate_history(frame)


def test_validation_rejects_duplicate_and_missing_week() -> None:
    duplicate = pd.concat([_history(), _history().iloc[[0]]], ignore_index=True)
    with pytest.raises(DataValidationError, match="duplicate"):
        validate_history(duplicate)
    gap = _history()
    gap.loc[1, "time_period"] = "2021-W02"
    with pytest.raises(DataValidationError, match="non-contiguous"):
        validate_history(gap)


def test_unknown_location_policy_is_explicit() -> None:
    with pytest.raises(DataValidationError, match="unknown location"):
        validate_history(
            _history(),
            ValidationConfig(
                known_locations=frozenset({"B"}),
                allow_unknown_locations=False,
            ),
        )
