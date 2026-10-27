from __future__ import annotations

import asyncio
import pickle

import pandas as pd
import pytest
from chapkit.data import DataFrame
from fastapi.testclient import TestClient
from geojson_pydantic import FeatureCollection

from chap_disease_forecasting.chapkit_adapter.service import (
    DiseaseModelConfig,
    create_app,
    on_predict,
    on_train,
)
from chap_disease_forecasting.data import add_iso_weeks
from chap_disease_forecasting.synthetic import (
    SyntheticConfig,
    generate_synthetic_history,
    synthetic_geojson,
)


def _small_history(n_locations: int = 2) -> pd.DataFrame:
    # Two backtest splits give n_locations * 2 residuals per horizon and
    # n_locations * 6 selection observations.
    return generate_synthetic_history(
        SyntheticConfig(
            n_locations=n_locations,
            n_weeks=36,
            missing_rate=0.0,
            short_history_fraction=0.0,
            seed=71,
        )
    )


def _future_from(history: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for location, group in history.groupby("location", observed=True):
        last = group.sort_values("time_period").iloc[-1]
        for horizon in (1, 2, 3):
            rows.append(
                {
                    "location": str(location),
                    "time_period": add_iso_weeks(last["time_period"], horizon),
                    "rainfall": float(last["rainfall"]),
                    "mean_temperature": float(last["mean_temperature"]),
                    "population": float(last["population"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def test_service_contract_exposes_health_info_and_config_schema(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'chapkit.db'}")
    application = create_app()

    with TestClient(application) as client:
        health = client.get("/health")
        info = client.get("/api/v1/info")
        schema = client.get("/api/v1/configs/$schema")

    assert health.status_code == 200
    assert info.status_code == 200
    assert info.json()["id"] == "chap-disease-forecasting"
    assert info.json()["period_type"] == "weekly"
    assert schema.status_code == 200
    engine_schema = schema.json()["properties"]["engine"]
    assert engine_schema["enum"] == ["xgboost", "lightgbm", "auto"]


def _small_config(engine: str) -> DiseaseModelConfig:
    return DiseaseModelConfig(
        engine=engine,
        n_estimators=10,
        n_threads=1,
        selection_backtest_splits=2,
        selection_min_train_weeks=26,
        selection_min_observations=20,
        prediction_samples=10,
        minimum_residuals_per_horizon=10,
    )


@pytest.mark.asyncio
async def test_callbacks_train_real_model_and_return_chap_samples(tmp_path, monkeypatch):
    # CHAPKit copies the working directory into every workspace zip, so the
    # callbacks must not write model files under it.
    monkeypatch.chdir(tmp_path)
    history = _small_history(n_locations=6)
    config = _small_config("xgboost")

    descriptor = await on_train(config, DataFrame.from_pandas(history))
    # CHAPKit persists the training result with pickle and reloads it for predict.
    restored = pickle.loads(pickle.dumps(descriptor))  # noqa: S301 - self-produced bytes
    prediction = await on_predict(
        config,
        restored,
        DataFrame.from_pandas(history),
        DataFrame.from_pandas(_future_from(history)),
    )
    output = prediction.to_pandas()

    assert list(tmp_path.iterdir()) == []
    assert descriptor["selection"]["policy"] == "manual"
    assert descriptor["models"].keys() == {"xgboost"}
    assert descriptor["uncertainty"]["calibrated"] is False
    assert len(output) == 18
    sample_columns = [column for column in output.columns if column.startswith("sample_")]
    assert sample_columns == [f"sample_{index}" for index in range(10)]
    assert output[sample_columns].notna().all().all()
    assert (output[sample_columns] >= 0).all().all()


@pytest.mark.asyncio
async def test_auto_engine_trains_both_and_records_selection():
    history = _small_history(n_locations=6)
    descriptor = await on_train(_small_config("auto"), DataFrame.from_pandas(history))

    selection = descriptor["selection"]
    assert descriptor["models"].keys() == {"xgboost", "lightgbm"}
    assert selection["policy"] == "auto_select"
    assert selection["reason"] in {
        "lowest eligible historical out-of-sample MAE",
        "incumbent retained to avoid a switch below the hysteresis margin",
    }
    assert {score["model_name"] for score in selection["scores"]} == {"xgboost", "lightgbm"}


@pytest.mark.asyncio
async def test_auto_engine_retains_champion_when_evidence_is_insufficient():
    history = _small_history(n_locations=2)
    config = _small_config("auto").model_copy(update={"champion": "lightgbm"})

    descriptor = await on_train(config, DataFrame.from_pandas(history))
    prediction = await on_predict(
        config,
        descriptor,
        DataFrame.from_pandas(history),
        DataFrame.from_pandas(_future_from(history)),
    )

    assert descriptor["selection"]["selected_models"] == ["lightgbm"]
    assert "insufficient historical out-of-sample evidence" in descriptor["selection"]["reason"]
    assert len(prediction.to_pandas()) == 6


@pytest.mark.asyncio
async def test_training_does_not_block_the_event_loop():
    # /health and the ServiceKit keepalive share the loop that awaits on_train;
    # Core drops a service whose keepalive is silent for 30 seconds.
    history = DataFrame.from_pandas(_small_history())
    done = asyncio.Event()
    ticks = 0

    async def ticker() -> None:
        nonlocal ticks
        while not done.is_set():
            await asyncio.sleep(0.01)
            if not done.is_set():
                ticks += 1

    async def train() -> None:
        await on_train(_small_config("xgboost"), history)
        done.set()

    await asyncio.gather(ticker(), train())

    assert ticks >= 3


@pytest.mark.asyncio
async def test_training_uses_chap_polygons_for_neighbour_features():
    config_data = SyntheticConfig(
        n_locations=6, n_weeks=36, missing_rate=0.0, short_history_fraction=0.0, seed=71
    )
    history = generate_synthetic_history(config_data)
    geo = FeatureCollection.model_validate(synthetic_geojson(config_data))

    descriptor = await on_train(_small_config("xgboost"), DataFrame.from_pandas(history), geo)
    # CHAP sends polygons again at predict time, but the graph stored at training suffices.
    prediction = await on_predict(
        _small_config("xgboost"),
        descriptor,
        DataFrame.from_pandas(history),
        DataFrame.from_pandas(_future_from(history)),
    )

    assert descriptor["spatial"]["locations_with_neighbours"] == 6
    assert len(prediction.to_pandas()) == 18


@pytest.mark.asyncio
async def test_predict_beyond_trained_horizon_asks_for_retraining():
    history = _small_history(n_locations=2)
    descriptor = await on_train(_small_config("xgboost"), DataFrame.from_pandas(history))
    future = _future_from(history)
    last = future.groupby("location").tail(1).copy()
    last["time_period"] = last["time_period"].map(lambda period: add_iso_weeks(period, 1))

    with pytest.raises(ValueError, match="trained for 3 week"):
        await on_predict(
            _small_config("xgboost"),
            descriptor,
            DataFrame.from_pandas(history),
            DataFrame.from_pandas(pd.concat([future, last], ignore_index=True)),
        )


@pytest.mark.asyncio
async def test_predict_rejects_descriptor_from_previous_format():
    history = DataFrame.from_pandas(_small_history())
    legacy = {"format_version": 1, "service_id": "chap-disease-forecasting", "artifacts": {}}

    with pytest.raises(ValueError, match="retrain"):
        await on_predict(_small_config("xgboost"), legacy, history, history)


def test_values_sent_to_chap_never_need_exponent_notation():
    # DHIS2 rejected a whole prediction import over one value, 6.584875126557921e-7
    # (E8122 value_not_numeric): JavaScript writes numbers below 1e-6 as "6.58e-7".
    from chap_disease_forecasting.chapkit_adapter.service import _align_to_future

    future = pd.DataFrame({"location": ["A", "B", "C"], "time_period": ["2026-W40"] * 3})
    predictions = pd.DataFrame(
        {
            "location": ["A", "B", "C"],
            "time_period": ["2026-W40"] * 3,
            "sample_0": [6.584875126557921e-7, 0.004, 12.3456],
            "sample_1": [0.0049, 3.333333, 1e-12],
        }
    )
    output = _align_to_future(future, predictions)

    assert output["sample_0"].tolist() == [0.0, 0.0, 12.35]
    assert output["sample_1"].tolist() == [0.0, 3.33, 0.0]
    values = output[["sample_0", "sample_1"]].to_numpy().ravel()
    assert not ((values > 0) & (values < 0.01)).any()
