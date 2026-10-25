from __future__ import annotations

import numpy as np

from chap_disease_forecasting.evaluation.comparison import (
    DEFAULT_VARIANTS,
    LEGACY_FEATURES,
    Variant,
    evaluate_variant,
)
from chap_disease_forecasting.features.spatial import neighbour_graph
from chap_disease_forecasting.synthetic import (
    SyntheticConfig,
    generate_synthetic_history,
    synthetic_geojson,
)


def test_variants_are_scored_out_of_sample_with_the_service_metrics() -> None:
    config = SyntheticConfig(
        n_locations=9, n_weeks=70, seed=3, disease="cholera", short_history_fraction=0.0
    )
    history = generate_synthetic_history(config)
    graph = neighbour_graph(synthetic_geojson(config), 4, sorted(history["location"].unique()))
    settings = dict(
        neighbours=graph,
        outer_splits=2,
        inner_splits=2,
        inner_min_train_weeks=40,
        n_estimators=10,
        n_samples=50,
        min_residuals_per_horizon=10,
    )
    rows = [evaluate_variant(history, variant, **settings) for variant in DEFAULT_VARIANTS[::2]]
    for row in rows:
        assert row["n_forecasts"] == 9 * 3 * 2
        assert 0 <= row["coverage_10_90"] <= 1 and 0 <= row["coverage_25_75"] <= 1
        assert np.isfinite([row["mae"], row["crps"], row["interval_score_10_90"]]).all()
        assert set(row["mae_by_horizon"]) == {1, 2, 3}
    # Variants are reproducible run to run.
    again = evaluate_variant(history, Variant("legacy", LEGACY_FEATURES, spatial=False), **settings)
    first = evaluate_variant(history, Variant("legacy", LEGACY_FEATURES, spatial=False), **settings)
    assert again["crps"] == first["crps"]
