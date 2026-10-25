"""One independently trained estimator per forecast horizon."""

from __future__ import annotations

import datetime as dt
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from chap_disease_forecasting.data import add_iso_weeks, validate_history
from chap_disease_forecasting.features import FeatureConfig, WeeklyFeatureBuilder
from chap_disease_forecasting.models.base import ModelNotFittedError, NativeRegressor
from chap_disease_forecasting.models.lightgbm import LightGBMRegressor
from chap_disease_forecasting.models.xgboost import XGBoostRegressor


@dataclass(frozen=True)
class DirectForecastConfig:
    horizons: tuple[int, ...] = (1, 2, 3)
    model_version: str = "0.1.0"

    def __post_init__(self) -> None:
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if len(set(self.horizons)) != len(self.horizons):
            raise ValueError("horizons must be unique")


class DirectForecaster:
    """Leakage-safe direct forecaster using independent horizon estimators."""

    def __init__(
        self,
        engine_factory: Callable[[], NativeRegressor],
        config: DirectForecastConfig | None = None,
        feature_config: FeatureConfig | None = None,
        neighbours: dict[str, list[str]] | None = None,
    ):
        self.engine_factory = engine_factory
        self.config = config or DirectForecastConfig()
        self.feature_builder = WeeklyFeatureBuilder(feature_config, neighbours)
        self.models_: dict[int, NativeRegressor] = {}
        self.trained_at_: str | None = None
        self.training_rows_: dict[int, int] = {}
        self.engine_name_: str | None = None

    def fit(self, history: pd.DataFrame) -> DirectForecaster:
        frame = validate_history(history)
        features = self.feature_builder.fit_transform(frame)
        targets = {
            horizon: frame.groupby("location", sort=False, observed=True)["disease_cases"].shift(
                -horizon
            )
            for horizon in self.config.horizons
        }
        models: dict[int, NativeRegressor] = {}
        training_rows: dict[int, int] = {}
        for horizon, target in targets.items():
            usable = target.notna() & np.isfinite(target.astype(float))
            if not usable.any():
                raise ValueError(f"no training targets available for horizon {horizon}")
            engine = self.engine_factory()
            engine.fit(
                features.loc[usable].to_numpy(dtype=np.float32),
                target.loc[usable].to_numpy(dtype=np.float32),
                list(features.columns),
            )
            models[horizon] = engine
            training_rows[horizon] = int(usable.sum())
        self.models_ = models
        self.training_rows_ = training_rows
        self.engine_name_ = next(iter(models.values())).engine_name
        self.trained_at_ = dt.datetime.now(dt.UTC).isoformat()
        return self

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if not self.models_ or self.trained_at_ is None or self.engine_name_ is None:
            raise ModelNotFittedError("DirectForecaster has not been fitted")
        frame = validate_history(history)
        features = self.feature_builder.transform(frame)
        latest_indices = (
            frame.groupby("location", sort=False, observed=True)["time_period"].idxmax().to_numpy()
        )
        latest = frame.loc[latest_indices, ["location", "time_period"]]
        latest_features = features.loc[latest_indices].to_numpy(dtype=np.float32)
        rows: list[dict[str, object]] = []
        for horizon in self.config.horizons:
            predictions = self.models_[horizon].predict(latest_features)
            if not np.isfinite(predictions).all():
                raise ValueError(f"non-finite predictions for horizon {horizon}")
            for (_, origin), prediction in zip(latest.iterrows(), predictions, strict=True):
                rows.append(
                    {
                        "location": str(origin["location"]),
                        "time_period": add_iso_weeks(origin["time_period"], horizon),
                        "horizon": horizon,
                        "prediction": max(float(prediction), 0.0),
                        "model_name": self.engine_name_,
                        "model_version": self.config.model_version,
                        "trained_at": self.trained_at_,
                        "used_fallback": False,
                    }
                )
        return pd.DataFrame.from_records(rows).sort_values(
            ["location", "horizon"], kind="stable", ignore_index=True
        )

    def save(self, directory: Path) -> None:
        if not self.models_ or self.trained_at_ is None or self.engine_name_ is None:
            raise ModelNotFittedError("DirectForecaster has not been fitted")
        directory.mkdir(parents=True, exist_ok=True)
        metadata = {
            "format_version": 1,
            "kind": "direct_forecaster",
            "engine": self.engine_name_,
            "config": {**asdict(self.config), "horizons": list(self.config.horizons)},
            "feature_config": self.feature_builder.config_dict,
            "location_mapping": self.feature_builder.location_mapping_,
            "parent_mapping": self.feature_builder.parent_mapping_,
            "neighbours": self.feature_builder.neighbours_,
            "feature_names": list(self.feature_builder.feature_names_ or ()),
            "time_origin": self.feature_builder.time_origin_.isoformat()
            if self.feature_builder.time_origin_ is not None
            else None,
            "trained_at": self.trained_at_,
            "training_rows": {str(key): value for key, value in self.training_rows_.items()},
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8"
        )
        for horizon, model in self.models_.items():
            model.save(directory / f"horizon_{horizon}")

    @classmethod
    def load(cls, directory: Path) -> DirectForecaster:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1 or metadata.get("kind") != "direct_forecaster":
            raise ValueError("incompatible direct forecaster artifact")
        engine_name = metadata.get("engine")
        engine_classes: dict[str, Any] = {
            "xgboost": XGBoostRegressor,
            "lightgbm": LightGBMRegressor,
        }
        if engine_name not in engine_classes:
            raise ValueError(f"unsupported engine in artifact: {engine_name!r}")
        config_data = dict(metadata["config"])
        config_data["horizons"] = tuple(config_data["horizons"])
        feature_data = dict(metadata["feature_config"])
        for key in (
            "case_lags",
            "case_windows",
            "climate_lags",
            "climate_windows",
            "climate_columns",
        ):
            feature_data[key] = tuple(feature_data[key])
        instance = cls(
            engine_factory=engine_classes[engine_name],
            config=DirectForecastConfig(**config_data),
            feature_config=FeatureConfig(**feature_data),
            neighbours={
                str(key): [str(value) for value in values]
                for key, values in metadata.get("neighbours", {}).items()
            },
        )
        instance.feature_builder.location_mapping_ = {
            str(key): int(value) for key, value in metadata["location_mapping"].items()
        }
        instance.feature_builder.parent_mapping_ = {
            str(key): int(value) for key, value in metadata["parent_mapping"].items()
        }
        instance.feature_builder.feature_names_ = tuple(metadata["feature_names"])
        instance.feature_builder.time_origin_ = pd.Timestamp(metadata["time_origin"])
        instance.models_ = {
            horizon: engine_classes[engine_name].load(directory / f"horizon_{horizon}")
            for horizon in instance.config.horizons
        }
        instance.trained_at_ = str(metadata["trained_at"])
        instance.training_rows_ = {
            int(key): int(value) for key, value in metadata["training_rows"].items()
        }
        instance.engine_name_ = engine_name
        return instance
