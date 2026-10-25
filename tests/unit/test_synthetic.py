from __future__ import annotations

from pandas.testing import assert_frame_equal

from chap_disease_forecasting.synthetic import generate_synthetic_history, preset_config


def test_synthetic_generator_is_reproducible_and_has_required_columns() -> None:
    config = preset_config(10, seed=123)
    first = generate_synthetic_history(config)
    second = generate_synthetic_history(config)
    assert_frame_equal(first, second)
    assert first["location"].nunique() == 10
    assert {
        "time_period",
        "location",
        "parent",
        "disease_cases",
        "rainfall",
        "mean_temperature",
        "population",
        "completeness",
    }.issubset(first.columns)


def test_required_scale_presets() -> None:
    assert (preset_config(10).n_locations, preset_config(10).n_weeks) == (10, 52)
    assert (preset_config(100).n_locations, preset_config(100).n_weeks) == (100, 196)
    assert (preset_config(514).n_locations, preset_config(514).n_weeks) == (514, 196)
