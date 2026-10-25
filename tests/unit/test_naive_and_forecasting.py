from __future__ import annotations

import pandas as pd

from chap_disease_forecasting.models import PersistenceForecaster, SeasonalNaiveForecaster
from chap_disease_forecasting.synthetic import SyntheticConfig, generate_synthetic_history


def _history() -> pd.DataFrame:
    return generate_synthetic_history(
        SyntheticConfig(
            n_locations=3,
            n_weeks=60,
            seed=11,
            missing_rate=0,
            short_history_fraction=0,
        )
    )


def test_naive_models_produce_three_non_negative_horizons() -> None:
    history = _history()
    for model in (PersistenceForecaster(), SeasonalNaiveForecaster()):
        output = model.fit(history).predict(history)
        assert set(output["horizon"]) == {1, 2, 3}
        assert len(output) == 9
        assert (output["prediction"] >= 0).all()
        assert not output.duplicated(["location", "time_period", "horizon"]).any()
