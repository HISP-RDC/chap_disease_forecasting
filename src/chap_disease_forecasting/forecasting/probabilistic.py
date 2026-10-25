"""Predictive samples from strictly out-of-sample historical residuals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd


class InsufficientResidualEvidence(ValueError):
    pass


_METHODS = {"additive": "empirical_oos_residual", "log1p": "empirical_oos_log_residual"}


@dataclass
class HorizonResidualSampler:
    """Sample horizon-specific backtest residuals around point forecasts.

    With ``scale="log1p"`` (the default) residuals are ``log1p(actual) -
    log1p(prediction)`` and samples are ``expm1(log1p(prediction) + residual)``:
    the spread grows with the forecast level, as it does for counts, instead of
    giving a zone with 5 cases the same absolute margin as one with 2,000.
    Residuals are further pooled by forecast level (``level_bins`` quantile
    bins of the transformed prediction, per horizon), because relative errors
    are larger for small counts; a horizon falls back to one pool when a bin
    would hold fewer than ``min_residuals_per_horizon`` residuals.

    Samples are labelled empirical rather than calibrated: finite rolling-origin
    residuals approximate uncertainty but do not, by themselves, prove
    calibration under temporal dependence.
    """

    n_samples: int = 100
    min_residuals_per_horizon: int = 20
    seed: int = 20261025
    scale: Literal["additive", "log1p"] = "log1p"
    level_bins: int = 3
    residuals_: dict[int, list[np.ndarray]] = field(default_factory=dict, init=False)
    edges_: dict[int, np.ndarray] = field(default_factory=dict, init=False)
    evaluated_through_: pd.Timestamp | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        if self.scale not in _METHODS:
            raise ValueError(f"unsupported residual scale: {self.scale!r}")
        if self.level_bins < 1:
            raise ValueError("level_bins must be at least 1")

    @property
    def method(self) -> str:
        return _METHODS[self.scale]

    def _transform(self, values: np.ndarray) -> np.ndarray:
        if self.scale == "log1p":
            return np.asarray(np.log1p(np.maximum(values, 0.0)), dtype=float)
        return values

    def _pools(
        self, level: np.ndarray, residual: np.ndarray
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        bins = self.level_bins
        while bins > 1:
            edges = np.unique(np.quantile(level, np.linspace(0, 1, bins + 1)[1:-1]))
            index = np.searchsorted(edges, level, side="right")
            pools = [residual[index == bin_index] for bin_index in range(len(edges) + 1)]
            if all(len(pool) >= self.min_residuals_per_horizon for pool in pools):
                return edges, pools
            bins -= 1
        return np.empty(0, dtype=float), [residual]

    def fit(self, backtest_results: pd.DataFrame) -> HorizonResidualSampler:
        required = {
            "actual",
            "prediction",
            "horizon",
            "forecast_origin",
            "is_out_of_sample",
        }
        missing = required.difference(backtest_results.columns)
        if missing:
            raise ValueError(f"backtest results missing columns: {sorted(missing)}")
        if not backtest_results["is_out_of_sample"].eq(True).all():  # noqa: E712
            raise ValueError("residual sampling requires exclusively out-of-sample predictions")
        residuals: dict[int, list[np.ndarray]] = {}
        edges: dict[int, np.ndarray] = {}
        for horizon, group in backtest_results.groupby("horizon", sort=True):
            horizon_number = int(str(horizon))
            actual = pd.to_numeric(group["actual"], errors="coerce").to_numpy(dtype=float)
            predicted = pd.to_numeric(group["prediction"], errors="coerce").to_numpy(dtype=float)
            usable = np.isfinite(actual) & np.isfinite(predicted)
            level = self._transform(predicted[usable])
            residual = self._transform(actual[usable]) - level
            if len(residual) < self.min_residuals_per_horizon:
                raise InsufficientResidualEvidence(
                    f"horizon {horizon_number} has {len(residual)} residuals; "
                    f"at least {self.min_residuals_per_horizon} required"
                )
            if np.unique(residual).size < 2:
                raise InsufficientResidualEvidence(
                    f"horizon {horizon_number} residual distribution has no variation"
                )
            edges[horizon_number], residuals[horizon_number] = self._pools(level, residual)
        if not residuals:
            raise InsufficientResidualEvidence("no usable horizon residuals")
        self.residuals_ = residuals
        self.edges_ = edges
        self.evaluated_through_ = pd.to_datetime(backtest_results["forecast_origin"]).max()
        return self

    def add_samples(self, point_forecasts: pd.DataFrame) -> pd.DataFrame:
        required = {"prediction", "horizon"}
        missing = required.difference(point_forecasts.columns)
        if missing:
            raise ValueError(f"point forecasts missing columns: {sorted(missing)}")
        if not self.residuals_ or self.evaluated_through_ is None:
            raise InsufficientResidualEvidence("residual sampler has not been fitted")
        if self.n_samples < 2:
            raise ValueError("at least two predictive samples are required")
        output = point_forecasts.reset_index(drop=True).copy()
        rng = np.random.default_rng(self.seed)
        prediction = output["prediction"].to_numpy(dtype=float)
        horizons = output["horizon"].to_numpy(dtype=int)
        samples = np.empty((len(output), self.n_samples), dtype=float)
        for horizon in sorted(set(horizons.tolist())):
            if horizon not in self.residuals_:
                raise InsufficientResidualEvidence(
                    f"no residual distribution for horizon {horizon}"
                )
            rows = np.flatnonzero(horizons == horizon)
            level = self._transform(prediction[rows])
            bin_index = np.searchsorted(self.edges_[horizon], level, side="right")
            for pool_index, pool in enumerate(self.residuals_[horizon]):
                selected = rows[bin_index == pool_index]
                if not len(selected):
                    continue
                draws = rng.choice(pool, size=(len(selected), self.n_samples), replace=True)
                centre = self._transform(prediction[selected])[:, None] + draws
                samples[selected] = np.expm1(centre) if self.scale == "log1p" else centre
        samples = np.maximum(samples, 0.0)
        sample_frame = pd.DataFrame(
            samples, columns=[f"sample_{index}" for index in range(self.n_samples)]
        )
        output = pd.concat([output, sample_frame], axis=1)
        output["probabilistic_method"] = self.method
        output["probabilistic_calibrated"] = False
        output["residuals_evaluated_through"] = self.evaluated_through_
        return output

    def to_dict(self) -> dict[str, Any]:
        if not self.residuals_ or self.evaluated_through_ is None:
            raise InsufficientResidualEvidence("residual sampler has not been fitted")
        return {
            "method": self.method,
            "scale": self.scale,
            "calibrated": False,
            "n_samples": self.n_samples,
            "seed": self.seed,
            "evaluated_through": self.evaluated_through_.isoformat(),
            "horizons": {
                str(horizon): {
                    "edges": self.edges_[horizon].astype(float).tolist(),
                    "pools": [pool.astype(float).tolist() for pool in pools],
                }
                for horizon, pools in self.residuals_.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HorizonResidualSampler:
        scale = data.get("scale")
        if scale not in _METHODS or data.get("method") != _METHODS[scale]:
            raise ValueError("unsupported uncertainty descriptor")
        sampler = cls(n_samples=int(data["n_samples"]), seed=int(data["seed"]), scale=scale)
        for horizon, entry in data["horizons"].items():
            edges = np.asarray(entry["edges"], dtype=float)
            pools = [np.asarray(pool, dtype=float) for pool in entry["pools"]]
            invalid_pool = any(
                pool.ndim != 1 or len(pool) < 2 or not np.isfinite(pool).all() for pool in pools
            )
            if len(pools) != len(edges) + 1 or not np.isfinite(edges).all() or invalid_pool:
                raise ValueError("invalid residual pools in uncertainty descriptor")
            sampler.residuals_[int(horizon)] = pools
            sampler.edges_[int(horizon)] = edges
        if not sampler.residuals_:
            raise ValueError("uncertainty descriptor has no horizons")
        sampler.evaluated_through_ = pd.Timestamp(data["evaluated_through"])
        return sampler


@dataclass
class ResidualSampleForecaster:
    point_forecaster: object
    sampler: HorizonResidualSampler

    def predict(self, history: pd.DataFrame) -> pd.DataFrame:
        if not hasattr(self.point_forecaster, "predict"):
            raise TypeError("point_forecaster must implement predict")
        return self.sampler.add_samples(self.point_forecaster.predict(history))
