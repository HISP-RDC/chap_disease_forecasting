from __future__ import annotations

import pandas as pd
from pandas.testing import assert_series_equal

from chap_disease_forecasting.features import WeeklyFeatureBuilder
from chap_disease_forecasting.synthetic import SyntheticConfig, generate_synthetic_history


def _history() -> pd.DataFrame:
    return generate_synthetic_history(
        SyntheticConfig(
            n_locations=2,
            n_weeks=60,
            seed=7,
            missing_rate=0,
            zero_inflation=0,
            short_history_fraction=0,
        )
    )


def test_required_features_exist_and_are_per_zone() -> None:
    history = _history()
    features = WeeklyFeatureBuilder().fit_transform(history)
    expected = {
        "cases_lag_1",
        "cases_lag_52",
        "cases_mean_4",
        "cases_mean_12",
        "cases_std_4",
        "incidence_lag_1_per_1000",
        "week_sin",
        "week_cos",
        "rainfall_lag_12",
        "mean_temperature_mean_8",
        "location_code",
    }
    assert expected.issubset(features.columns)
    second_zone_first = history.index[history["location"].eq("ZS-0002")][0]
    assert pd.isna(features.loc[second_zone_first, "cases_lag_1"])


def test_future_mutation_cannot_change_past_features() -> None:
    original = _history()
    changed = original.copy()
    cutoff = original["time_period"].sort_values().iloc[35]
    changed.loc[changed["time_period"] > cutoff, "disease_cases"] = 1_000_000
    builder = WeeklyFeatureBuilder().fit(original)
    baseline = builder.transform(original)
    mutated = builder.transform(changed)
    past = original["time_period"] <= cutoff
    assert_series_equal(baseline.loc[past, "cases_mean_12"], mutated.loc[past, "cases_mean_12"])


def test_zone_mutation_does_not_leak_between_locations() -> None:
    original = _history()
    changed = original.copy()
    changed.loc[changed["location"].eq("ZS-0002"), "disease_cases"] = 999_999
    builder = WeeklyFeatureBuilder().fit(original)
    baseline = builder.transform(original)
    mutated = builder.transform(changed)
    zone_a = original["location"].eq("ZS-0001")
    assert_series_equal(baseline.loc[zone_a, "cases_lag_1"], mutated.loc[zone_a, "cases_lag_1"])
