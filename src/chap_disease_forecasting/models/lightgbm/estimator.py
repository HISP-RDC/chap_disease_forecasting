"""CPU-bounded LightGBM count regressor with native text persistence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from chap_disease_forecasting.models.base import EngineUnavailableError, ModelNotFittedError


@dataclass(frozen=True)
class LightGBMConfig:
    n_estimators: int = 120
    learning_rate: float = 0.05
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.05
    reg_lambda: float = 1.0
    objective: str = "poisson"
    n_jobs: int = 1
    random_state: int = 20261025
    # Only for a Tweedie objective; left out of the engine parameters otherwise.
    tweedie_variance_power: float | None = None
    verbosity: int = -1


def _engine_params(config: LightGBMConfig) -> dict[str, Any]:
    return {key: value for key, value in asdict(config).items() if value is not None}


class LightGBMRegressor:
    engine_name = "lightgbm"

    def __init__(self, config: LightGBMConfig | None = None):
        self.config = config or LightGBMConfig()
        self.model_: Any | None = None
        self.feature_names_: tuple[str, ...] | None = None

    @staticmethod
    def _library() -> Any:
        try:
            import lightgbm as lgb
        except ImportError as exc:
            raise EngineUnavailableError("lightgbm is not installed") from exc
        return lgb

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> LightGBMRegressor:
        lgb = self._library()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or len(y) != len(X):
            raise ValueError("X must be 2-D and aligned with y")
        if len(y) == 0 or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("y must contain finite non-negative observations")
        if X.shape[1] != len(feature_names):
            raise ValueError("feature_names does not match X")
        model = lgb.LGBMRegressor(**_engine_params(self.config))
        model.fit(X, y, feature_name=list(feature_names))
        self.model_ = model
        self.feature_names_ = tuple(feature_names)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ModelNotFittedError("LightGBM model has not been fitted")
        predictor = getattr(self.model_, "booster_", self.model_)
        values: np.ndarray = np.asarray(
            predictor.predict(np.asarray(X, dtype=np.float32)), dtype=float
        )
        return cast(np.ndarray, np.maximum(values, 0.0))

    def save(self, directory: Path) -> None:
        if self.model_ is None or self.feature_names_ is None:
            raise ModelNotFittedError("LightGBM model has not been fitted")
        directory.mkdir(parents=True, exist_ok=True)
        predictor = getattr(self.model_, "booster_", self.model_)
        predictor.save_model(str(directory / "model.txt"))
        metadata = {
            "format_version": 1,
            "engine": self.engine_name,
            "library_version": self._library().__version__,
            "config": asdict(self.config),
            "feature_names": list(self.feature_names_),
        }
        (directory / "metadata.json").write_text(
            json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: Path) -> LightGBMRegressor:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1 or metadata.get("engine") != cls.engine_name:
            raise ValueError("incompatible LightGBM artifact metadata")
        instance = cls(LightGBMConfig(**metadata["config"]))
        lgb = instance._library()
        instance.model_ = lgb.Booster(model_file=str(directory / "model.txt"))
        instance.feature_names_ = tuple(metadata["feature_names"])
        return instance
