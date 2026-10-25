from __future__ import annotations

import pandas as pd
import pytest

from chap_disease_forecasting.models.ensemble import (
    AutoSelectionPolicy,
    ChampionChallengerPolicy,
    EnsembleForecaster,
    FallbackForecaster,
    ForecastFailure,
    HistoricalScore,
)


class StaticModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if self.value < 0:
            raise RuntimeError("engine failed")
        return pd.DataFrame(
            {
                "location": ["A"],
                "time_period": [pd.Timestamp("2026-01-05")],
                "horizon": [1],
                "prediction": [self.value],
            }
        )


def _scores() -> list[HistoricalScore]:
    return [
        HistoricalScore("xgboost", 4.0, 200, pd.Timestamp("2025-12-01")),
        HistoricalScore("lightgbm", 3.0, 200, pd.Timestamp("2025-12-01")),
        HistoricalScore("future", 0.1, 200, pd.Timestamp("2027-01-01")),
    ]


def test_auto_selection_ignores_future_evidence_and_records_reason() -> None:
    decision = AutoSelectionPolicy().select(_scores(), pd.Timestamp("2026-01-01"))
    assert decision.selected_models == ("lightgbm",)
    assert {score.model_name for score in decision.scores} == {"xgboost", "lightgbm"}
    assert decision.reason


def test_challenger_requires_promotion_margin() -> None:
    decision = ChampionChallengerPolicy(
        champion="xgboost", challenger="lightgbm", promotion_margin=0.10
    ).select(_scores(), pd.Timestamp("2026-01-01"))
    assert decision.selected_models == ("lightgbm",)


def test_weighted_ensemble_uses_historical_weights() -> None:
    model = EnsembleForecaster(
        models={"a": StaticModel(10), "b": StaticModel(20)}, weights={"a": 0.25, "b": 0.75}
    )
    output = model.predict(pd.DataFrame())
    assert output.loc[0, "prediction"] == 17.5
    assert output.loc[0, "ensemble_members"] == "a,b"


def test_fallback_is_traceable_and_total_failure_is_structured() -> None:
    model = FallbackForecaster(
        models={"xgboost": StaticModel(-1), "lightgbm": StaticModel(8)},
        primary="xgboost",
        alternatives=("lightgbm",),
    )
    output = model.predict(pd.DataFrame())
    assert bool(output.loc[0, "used_fallback"])
    assert output.loc[0, "selected_model"] == "lightgbm"
    assert "engine failed" in output.loc[0, "fallback_reason"]
    with pytest.raises(ForecastFailure) as error:
        FallbackForecaster(
            models={"xgboost": StaticModel(-1)},
            primary="xgboost",
            alternatives=("missing",),
        ).predict(pd.DataFrame())
    assert set(error.value.attempts) == {"xgboost", "missing"}
