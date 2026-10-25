"""Reproducible synthetic fixtures; not scientifically validated data."""

from .generator import (
    SyntheticConfig,
    generate_synthetic_history,
    preset_config,
    synthetic_geojson,
)

__all__ = ["SyntheticConfig", "generate_synthetic_history", "preset_config", "synthetic_geojson"]
