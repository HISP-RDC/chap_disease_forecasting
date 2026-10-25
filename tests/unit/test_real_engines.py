from __future__ import annotations

import copy

import pytest

from chap_disease_forecasting.artifacts import (
    ArtifactIntegrityError,
    ArtifactStore,
    pack_forecaster,
    unpack_forecaster,
)
from chap_disease_forecasting.forecasting import DirectForecaster
from chap_disease_forecasting.models.lightgbm import LightGBMConfig, LightGBMRegressor
from chap_disease_forecasting.models.xgboost import XGBoostConfig, XGBoostRegressor
from chap_disease_forecasting.synthetic import SyntheticConfig, generate_synthetic_history


@pytest.fixture(scope="module")
def history():
    return generate_synthetic_history(
        SyntheticConfig(
            n_locations=3,
            n_weeks=60,
            seed=19,
            missing_rate=0,
            short_history_fraction=0,
        )
    )


@pytest.mark.parametrize("engine", ["xgboost", "lightgbm"])
def test_real_engine_trains_three_horizons_and_is_deterministic(engine: str, history) -> None:
    pytest.importorskip(engine)
    if engine == "xgboost":

        def factory():
            return XGBoostRegressor(XGBoostConfig(n_estimators=8, n_jobs=1))

    else:

        def factory():
            return LightGBMRegressor(LightGBMConfig(n_estimators=8, n_jobs=1))

    first = DirectForecaster(factory).fit(history).predict(history)
    second = DirectForecaster(factory).fit(history).predict(history)
    assert len(first) == 3 * history["location"].nunique()
    assert set(first["horizon"]) == {1, 2, 3}
    assert (first["prediction"] >= 0).all()
    assert first["prediction"].to_numpy() == pytest.approx(
        second["prediction"].to_numpy(), rel=1e-6, abs=1e-6
    )


@pytest.mark.parametrize("engine", ["xgboost", "lightgbm"])
def test_native_artifact_roundtrip_and_tamper_detection(tmp_path, history, engine) -> None:
    pytest.importorskip(engine)

    if engine == "xgboost":

        def factory():
            return XGBoostRegressor(XGBoostConfig(n_estimators=5, n_jobs=1))

    else:

        def factory():
            return LightGBMRegressor(LightGBMConfig(n_estimators=5, n_jobs=1))

    fitted = DirectForecaster(factory).fit(history)
    store = ArtifactStore(tmp_path)
    artifact_id = f"{engine}-v1"
    path = store.save(artifact_id, fitted)
    restored = store.load(artifact_id)
    assert restored.predict(history)["prediction"].to_numpy() == pytest.approx(
        fitted.predict(history)["prediction"].to_numpy()
    )
    with pytest.raises(ValueError, match="unsafe"):
        store.save("../escape", fitted)
    metadata = path / "metadata.json"
    metadata.write_text(metadata.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        store.load(artifact_id)


@pytest.mark.parametrize("engine", ["xgboost", "lightgbm"])
def test_in_memory_bundle_roundtrip_and_rejections(history, engine) -> None:
    pytest.importorskip(engine)
    if engine == "xgboost":

        def factory():
            return XGBoostRegressor(XGBoostConfig(n_estimators=5, n_jobs=1))

    else:

        def factory():
            return LightGBMRegressor(LightGBMConfig(n_estimators=5, n_jobs=1))

    fitted = DirectForecaster(factory).fit(history)

    bundle = pack_forecaster(fitted)
    restored = unpack_forecaster(copy.deepcopy(bundle))
    assert restored.predict(history)["prediction"].to_numpy() == pytest.approx(
        fitted.predict(history)["prediction"].to_numpy()
    )

    tampered = copy.deepcopy(bundle)
    tampered["files"]["metadata.json"]["text"] += "\n"
    with pytest.raises(ArtifactIntegrityError, match="checksum"):
        unpack_forecaster(tampered)

    escaping = copy.deepcopy(bundle)
    escaping["files"]["../escape.json"] = escaping["files"].pop("metadata.json")
    with pytest.raises(ArtifactIntegrityError, match="unsafe"):
        unpack_forecaster(escaping)
