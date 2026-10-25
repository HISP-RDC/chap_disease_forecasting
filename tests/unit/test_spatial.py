from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from chap_disease_forecasting.artifacts import pack_forecaster, unpack_forecaster
from chap_disease_forecasting.features import FeatureConfig, WeeklyFeatureBuilder
from chap_disease_forecasting.features.spatial import (
    geometry_centroid,
    location_centroids,
    nearest_neighbours,
    neighbour_graph,
)
from chap_disease_forecasting.forecasting import DirectForecaster
from chap_disease_forecasting.models.xgboost import XGBoostConfig, XGBoostRegressor
from chap_disease_forecasting.synthetic import (
    SyntheticConfig,
    generate_synthetic_history,
    synthetic_geojson,
)

NEIGHBOUR_COLUMNS = [
    "neighbours_incidence",
    "neighbours_incidence_mean_4",
    "neighbours_log_growth_4w",
    "neighbours_pressure",
    "neighbours_missing",
]


def _config() -> SyntheticConfig:
    return SyntheticConfig(n_locations=9, n_weeks=60, seed=5, disease="cholera", missing_rate=0.0)


def _graph(config: SyntheticConfig, history: pd.DataFrame) -> dict[str, list[str]]:
    return neighbour_graph(synthetic_geojson(config), 4, sorted(history["location"].unique()))


def test_centroids_of_polygons_and_multipolygons() -> None:
    square = [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0], [0.0, 0.0]]
    assert geometry_centroid({"type": "Polygon", "coordinates": [square]}) == pytest.approx(
        (1.0, 1.0)
    )
    far = [[10.0, 0.0], [12.0, 0.0], [12.0, 2.0], [10.0, 2.0], [10.0, 0.0]]
    centroid = geometry_centroid({"type": "MultiPolygon", "coordinates": [[square], [far]]})
    assert centroid == pytest.approx((6.0, 1.0))
    collection = {
        "type": "FeatureCollection",
        "features": [
            {"type": "Feature", "id": "A", "geometry": {"type": "Point", "coordinates": [1, 2]}},
            {
                "type": "Feature",
                "properties": {"id": "B"},
                "geometry": {"type": "Polygon", "coordinates": [square]},
            },
            {"type": "Feature", "properties": {}, "geometry": None},
        ],
    }
    assert location_centroids(collection) == {"A": (1.0, 2.0), "B": pytest.approx((1.0, 1.0))}


def test_nearest_neighbours_on_a_grid() -> None:
    config = _config()
    graph = neighbour_graph(synthetic_geojson(config), 4, [f"ZS-{i:04d}" for i in range(1, 10)])
    # 3x3 grid: the centre cell's four nearest cells are its edge neighbours.
    assert sorted(graph["ZS-0005"]) == ["ZS-0002", "ZS-0004", "ZS-0006", "ZS-0008"]
    assert "ZS-0005" not in graph["ZS-0005"]
    assert nearest_neighbours({"A": (0.0, 0.0)}, 3) == {}
    assert neighbour_graph(None, 4, ["A", "B"]) == {}


def test_neighbour_features_only_use_the_neighbours_past() -> None:
    config = _config()
    history = generate_synthetic_history(config)
    graph = _graph(config, history)
    cutoff = history["time_period"].sort_values().unique()[40]
    changed = history.copy()
    changed.loc[changed["time_period"] > cutoff, "disease_cases"] = 1_000_000
    builder = WeeklyFeatureBuilder(neighbours=graph).fit(history)
    past = history["time_period"] <= cutoff
    assert_frame_equal(
        builder.transform(history).loc[past, NEIGHBOUR_COLUMNS],
        builder.transform(changed).loc[past, NEIGHBOUR_COLUMNS],
    )


def test_an_outbreak_next_door_raises_the_neighbour_signal() -> None:
    config = _config()
    history = generate_synthetic_history(config)
    graph = _graph(config, history)
    week = history["time_period"].sort_values().unique()[45]
    surge = history.copy()
    surge.loc[surge["location"].eq("ZS-0002") & surge["time_period"].eq(week), "disease_cases"] = (
        5_000
    )
    builder = WeeklyFeatureBuilder(neighbours=graph).fit(history)
    row = history.index[history["location"].eq("ZS-0005") & history["time_period"].eq(week)][0]
    before = builder.transform(history).loc[row, "neighbours_incidence"]
    after = builder.transform(surge).loc[row, "neighbours_incidence"]
    assert after > before
    # A location that is not a neighbour of ZS-0002 is unaffected.
    other = history.index[history["location"].eq("ZS-0009") & history["time_period"].eq(week)][0]
    assert builder.transform(surge).loc[other, "neighbours_incidence"] == pytest.approx(
        builder.transform(history).loc[other, "neighbours_incidence"]
    )


def test_no_graph_or_disabled_spatial_means_no_neighbour_columns() -> None:
    history = generate_synthetic_history(_config())
    graph = _graph(_config(), history)
    plain = WeeklyFeatureBuilder().fit_transform(history)
    disabled = WeeklyFeatureBuilder(FeatureConfig(spatial=False), graph).fit_transform(history)
    assert not set(NEIGHBOUR_COLUMNS) & set(plain.columns)
    assert not set(NEIGHBOUR_COLUMNS) & set(disabled.columns)


def test_dynamics_and_climate_features_exist() -> None:
    features = WeeklyFeatureBuilder().fit_transform(generate_synthetic_history(_config()))
    expected = {
        "log_growth_1w",
        "log_growth_4w",
        "incidence_at_origin_per_1000",
        "rainfall_lag_0",
        "rainfall_mean_16",
        "rainfall_anomaly_4",
        "mean_temperature_anomaly_4",
    }
    assert expected <= set(features.columns)
    assert np.isfinite(features["log_growth_4w"].dropna()).all()


def test_neighbour_graph_survives_packing_and_needs_no_geometry_to_predict() -> None:
    config = _config()
    history = generate_synthetic_history(config)
    graph = _graph(config, history)
    fitted = DirectForecaster(
        lambda: XGBoostRegressor(XGBoostConfig(n_estimators=5)), neighbours=graph
    ).fit(history)
    restored = unpack_forecaster(pack_forecaster(fitted))
    assert restored.feature_builder.neighbours_ == graph
    assert "neighbours_incidence" in (restored.feature_builder.feature_names_ or ())
    np.testing.assert_allclose(
        restored.predict(history)["prediction"], fitted.predict(history)["prediction"]
    )
