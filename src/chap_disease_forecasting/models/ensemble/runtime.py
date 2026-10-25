"""Prediction combination and error-aware fallback orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import pandas as pd


class Forecaster(Protocol):
    def predict(self, history: pd.DataFrame) -> pd.DataFrame: ...


class ForecastFailure(RuntimeError):
    def __init__(self, attempts: dict[str, str]):
        self.attempts = attempts
        super().__init__(f"all forecast attempts failed: {attempts}")


def _validate_output(output: pd.DataFrame) -> pd.DataFrame:
    required = {"location", "time_period", "horizon", "prediction"}
    missing = required.difference(output.columns)
    if missing:
        raise ValueError(f"prediction output missing columns: {sorted(missing)}")
    if output.empty:
        raise ValueError("prediction output is empty")
    prediction = pd.to_numeric(output["prediction"], errors="coerce")
    if prediction.isna().any() or not np.isfinite(prediction).all() or (prediction < 0).any():
        raise ValueError("predictions must be finite and non-negative")
    if output.duplicated(["location", "time_period", "horizon"]).any():
        raise ValueError("duplicate prediction keys")
    return output.copy()


@dataclass
class EnsembleForecaster:
    models: dict[str, Forecaster]
    weights: dict[str, float]
    model_name: str = "weighted_ensemble"

    def __post_init__(self) -> None:
        if set(self.models) != set(self.weights):
            raise ValueError("models and weights must have identical names")
        if not self.weights or any(weight < 0 for weight in self.weights.values()):
            raise ValueError("ensemble weights must be non-negative")
        total = sum(self.weights.values())
        if total <= 0:
            raise ValueError("at least one ensemble weight must be positive")
        self.weights = {name: weight / total for name, weight in self.weights.items()}

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        keys = ["location", "time_period", "horizon"]
        combined: pd.DataFrame | None = None
        for name, model in self.models.items():
            output = _validate_output(model.predict(history))
            current = output[keys + ["prediction"]].rename(
                columns={"prediction": f"prediction_{name}"}
            )
            combined = (
                current if combined is None else combined.merge(current, on=keys, how="inner")
            )
        assert combined is not None
        if combined.empty:
            raise ValueError("ensemble members do not share prediction keys")
        combined["prediction"] = sum(
            combined[f"prediction_{name}"] * weight for name, weight in self.weights.items()
        )
        combined["model_name"] = self.model_name
        combined["ensemble_members"] = ",".join(self.models)
        combined["ensemble_weights"] = ",".join(
            f"{name}:{self.weights[name]:.12g}" for name in self.models
        )
        combined["used_fallback"] = False
        return combined[
            keys
            + ["prediction", "model_name", "ensemble_members", "ensemble_weights", "used_fallback"]
        ]


@dataclass
class FallbackForecaster:
    """Try configured trained models in order and record every fallback."""

    models: dict[str, Forecaster]
    primary: str
    alternatives: tuple[str, ...] = ()

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        order = (self.primary,) + self.alternatives
        attempts: dict[str, str] = {}
        for index, name in enumerate(order):
            if name not in self.models:
                attempts[name] = "model is unavailable"
                continue
            try:
                output = _validate_output(self.models[name].predict(history))
            except Exception as exc:  # fallback boundary intentionally catches engine failures
                attempts[name] = f"{type(exc).__name__}: {exc}"
                continue
            output["requested_model"] = self.primary
            output["selected_model"] = name
            output["used_fallback"] = index > 0
            output["fallback_reason"] = (
                ""
                if index == 0
                else "; ".join(f"{failed}: {reason}" for failed, reason in attempts.items())
            )
            return output
        raise ForecastFailure(attempts)
