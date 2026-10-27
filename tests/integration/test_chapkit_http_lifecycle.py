"""Full CHAPKit HTTP lifecycle, driven the way CHAP Core 2.4.0 drives a service.

Payload construction mirrors ``chap_core/models/chapkit_rest_api_wrapper.py`` and
``external_chapkit_model.py`` at tag v2.4.0: flat user options under ``data``,
``time_period`` as ``YYYY-Wnn`` strings, ``NaN`` sent as ``null``, job polling
until a terminal status, then ``MLPredictionArtifactData`` validation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd
import pytest
from chapkit.artifact.schemas import MLPredictionArtifactData
from chapkit.data import DataFrame
from fastapi.testclient import TestClient

from chap_disease_forecasting.chapkit_adapter.service import create_app
from chap_disease_forecasting.data import add_iso_weeks
from chap_disease_forecasting.data.weeks import iso_week_label
from chap_disease_forecasting.synthetic import (
    SyntheticConfig,
    generate_synthetic_history,
    synthetic_geojson,
)

pytestmark = pytest.mark.integration

TERMINAL = {"completed", "failed", "canceled"}


def _prepare_dataframe(frame: pd.DataFrame) -> dict[str, Any]:
    frame = frame.copy()
    frame["time_period"] = frame["time_period"].map(iso_week_label)
    frame = frame.astype(object).replace({np.nan: None})
    return {"columns": frame.columns.tolist(), "data": frame.values.tolist()}


HISTORY_CONFIG = SyntheticConfig(
    n_locations=6, n_weeks=60, missing_rate=0.02, short_history_fraction=0.0, seed=2024
)
# CHAP sends the dataset polygons with both $train and $predict.
GEO = synthetic_geojson(HISTORY_CONFIG)


def _history() -> pd.DataFrame:
    return generate_synthetic_history(HISTORY_CONFIG)


def _future(history: pd.DataFrame, periods: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for location, group in history.groupby("location", observed=True):
        last = group.sort_values("time_period").iloc[-1]
        for horizon in range(1, periods + 1):
            rows.append(
                {
                    "location": str(location),
                    "time_period": add_iso_weeks(last["time_period"], horizon),
                    "rainfall": float(group["rainfall"].mean()),
                    "mean_temperature": float(group["mean_temperature"].mean()),
                    "population": float(last["population"]),
                }
            )
    return pd.DataFrame.from_records(rows)


def _wait(client: TestClient, job_id: str, timeout: float = 120.0) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.get(f"/api/v1/jobs/{job_id}").json()
        if job["status"] in TERMINAL:
            return job
        time.sleep(0.1)
    raise TimeoutError(job_id)


def _predict(
    client: TestClient, artifact_id: str, history: pd.DataFrame, periods: int
) -> pd.DataFrame:
    response = client.post(
        "/api/v1/ml/$predict",
        json={
            "artifact_id": artifact_id,
            "historic": _prepare_dataframe(history),
            "future": _prepare_dataframe(_future(history, periods)),
            "run_info": {"prediction_periods": periods},
            "geo": GEO,
        },
    )
    assert response.status_code == 202, response.text
    job = _wait(client, response.json()["job_id"])
    assert job["status"] == "completed", job
    artifact = client.get(f"/api/v1/artifacts/{response.json()['artifact_id']}").json()
    data = MLPredictionArtifactData.model_validate(artifact["data"])
    return DataFrame(**data.content).to_pandas()


# 12 is what CHAP Core 2.4.0 requests for a weekly prediction without an explicit
# horizon (db_worker_functions._get_n_periods); 3 is the usual backtest length.
@pytest.mark.parametrize(
    ("engine", "periods"), [("xgboost", 3), ("lightgbm", 3), ("auto", 3), ("auto", 12)]
)
def test_core_style_train_predict_and_restart(tmp_path, monkeypatch, engine, periods):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'chapkit.db'}")
    monkeypatch.setenv("CHAPKIT_TEMP_DIR", str(tmp_path))
    history = _history()
    sent = {
        "engine": engine,
        "n_estimators": 10,
        # Three splits keep >= 10 residuals for each of 12 horizons despite missing cases.
        "selection_backtest_splits": 3,
        "selection_min_train_weeks": 26,
        "selection_min_observations": 20,
        "prediction_samples": 20,
        "minimum_residuals_per_horizon": 10,
    }

    with TestClient(create_app()) as client:
        created = client.post("/api/v1/configs", json={"name": f"{engine}_1", "data": sent})
        assert created.status_code == 201, created.text
        stored = created.json()["data"]
        assert {key: stored[key] for key in sent} == sent

        train = client.post(
            "/api/v1/ml/$train",
            json={
                "config_id": created.json()["id"],
                "data": _prepare_dataframe(history),
                "run_info": {"prediction_periods": periods},
                "geo": GEO,
            },
        )
        assert train.status_code == 202, train.text
        job = _wait(client, train.json()["job_id"])
        assert job["status"] == "completed", job
        training_artifact = train.json()["artifact_id"]
        first = _predict(client, training_artifact, history, periods)

    # A new process on the same database must still predict from the old training.
    with TestClient(create_app()) as client:
        assert client.get("/health").json()["status"] == "healthy"
        second = _predict(client, training_artifact, history, periods)

    samples = [column for column in first.columns if column.startswith("sample_")]
    assert samples == [f"sample_{index}" for index in range(20)]
    assert len(first) == history["location"].nunique() * periods
    assert set(first["location"]) == set(history["location"].astype(str))
    assert np.isfinite(first[samples].to_numpy(dtype=float)).all()
    assert (first[samples].to_numpy(dtype=float) >= 0).all()
    # Hundredths of a case at most, so DHIS2 never receives exponent notation.
    np.testing.assert_array_equal(
        first[samples].to_numpy(dtype=float), first[samples].to_numpy(dtype=float).round(2)
    )
    pd.testing.assert_frame_equal(first, second)
