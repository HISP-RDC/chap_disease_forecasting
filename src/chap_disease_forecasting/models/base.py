"""Small engine interface shared by XGBoost and LightGBM."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


class ModelNotFittedError(RuntimeError):
    pass


class EngineUnavailableError(ImportError):
    pass


@runtime_checkable
class NativeRegressor(Protocol):
    engine_name: str
    feature_names_: tuple[str, ...] | None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Sequence[str],
    ) -> NativeRegressor: ...

    def predict(self, X: np.ndarray) -> np.ndarray: ...

    def save(self, directory: Path) -> None: ...
