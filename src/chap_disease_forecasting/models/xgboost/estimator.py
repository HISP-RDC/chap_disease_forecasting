"""CPU-bounded XGBoost count regressor with native JSON persistence."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from chap_disease_forecasting.models.base import EngineUnavailableError, ModelNotFittedError


@dataclass(frozen=True)
class XGBoostConfig:
    n_estimators: int = 120
    learning_rate: float = 0.05
    max_depth: int = 5
    min_child_weight: float = 2.0
    subsample: float = 0.85
    colsample_bytree: float = 0.85
    reg_alpha: float = 0.05
    reg_lambda: float = 1.0
    objective: str = "count:poisson"
    tree_method: str = "hist"
    n_jobs: int = 1
    random_state: int = 20261025
    # Only for a Tweedie objective; left out of the engine parameters otherwise.
    tweedie_variance_power: float | None = None


def _engine_params(config: XGBoostConfig) -> dict[str, Any]:
    return {key: value for key, value in asdict(config).items() if value is not None}


class XGBoostRegressor:
    engine_name = "xgboost"

    def __init__(self, config: XGBoostConfig | None = None):
        self.config = config or XGBoostConfig()
        self.model_: Any | None = None
        self.feature_names_: tuple[str, ...] | None = None

    @staticmethod
    def _library() -> Any:
        try:
            import xgboost as xgb
        except ImportError as exc:
            raise EngineUnavailableError("xgboost is not installed") from exc
        return xgb

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> XGBoostRegressor:
        xgb = self._library()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        if X.ndim != 2 or len(y) != len(X):
            raise ValueError("X must be 2-D and aligned with y")
        if len(y) == 0 or not np.isfinite(y).all() or (y < 0).any():
            raise ValueError("y must contain finite non-negative observations")
        if X.shape[1] != len(feature_names):
            raise ValueError("feature_names does not match X")
        model = xgb.XGBRegressor(**_engine_params(self.config))
        model.fit(X, y, verbose=False)
        self.model_ = model
        self.feature_names_ = tuple(feature_names)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise ModelNotFittedError("XGBoost model has not been fitted")
        values: np.ndarray = np.asarray(
            self.model_.predict(np.asarray(X, dtype=np.float32)), dtype=float
        )
        return cast(np.ndarray, np.maximum(values, 0.0))

    def save(self, directory: Path) -> None:
        if self.model_ is None or self.feature_names_ is None:
            raise ModelNotFittedError("XGBoost model has not been fitted")
        directory.mkdir(parents=True, exist_ok=True)
        self.model_.save_model(directory / "model.json")
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
    def load(cls, directory: Path) -> XGBoostRegressor:
        metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
        if metadata.get("format_version") != 1 or metadata.get("engine") != cls.engine_name:
            raise ValueError("incompatible XGBoost artifact metadata")
        instance = cls(XGBoostConfig(**metadata["config"]))
        xgb = instance._library()
        model = xgb.XGBRegressor(**_engine_params(instance.config))
        model.load_model(directory / "model.json")
        instance.model_ = model
        instance.feature_names_ = tuple(metadata["feature_names"])
        return instance
