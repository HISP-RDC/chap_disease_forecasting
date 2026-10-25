"""Selection policies based only on already-observed validation results."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class InsufficientEvidenceError(ValueError):
    """No candidate has enough already-observed validation results to be ranked."""


@dataclass(frozen=True)
class HistoricalScore:
    model_name: str
    mae: float
    n_observations: int
    evaluated_through: pd.Timestamp
    scope: str = "global"

    def __post_init__(self) -> None:
        if self.mae < 0 or self.n_observations < 0:
            raise ValueError("historical scores must be non-negative")


@dataclass(frozen=True)
class SelectionDecision:
    policy: str
    selected_models: tuple[str, ...]
    weights: tuple[float, ...]
    reason: str
    scores: tuple[HistoricalScore, ...]


def _eligible(
    scores: list[HistoricalScore], forecast_origin: pd.Timestamp, minimum_observations: int
) -> list[HistoricalScore]:
    origin = pd.Timestamp(forecast_origin)
    return [
        score
        for score in scores
        if score.n_observations >= minimum_observations
        and pd.Timestamp(score.evaluated_through) <= origin
    ]


@dataclass(frozen=True)
class ManualSelectionPolicy:
    model_name: str

    def select(
        self, scores: list[HistoricalScore], forecast_origin: pd.Timestamp
    ) -> SelectionDecision:
        known = tuple(score for score in scores if score.model_name == self.model_name)
        return SelectionDecision(
            policy="manual",
            selected_models=(self.model_name,),
            weights=(1.0,),
            reason="model explicitly selected by configuration",
            scores=known,
        )


@dataclass(frozen=True)
class ChampionChallengerPolicy:
    champion: str
    challenger: str
    minimum_observations: int = 100
    promotion_margin: float = 0.05

    def select(
        self, scores: list[HistoricalScore], forecast_origin: pd.Timestamp
    ) -> SelectionDecision:
        eligible = _eligible(scores, forecast_origin, self.minimum_observations)
        by_name = {score.model_name: score for score in eligible}
        champion_score = by_name.get(self.champion)
        challenger_score = by_name.get(self.challenger)
        if champion_score is None or challenger_score is None:
            selected = self.champion
            reason = "insufficient comparable historical observations; champion retained"
        elif challenger_score.mae < champion_score.mae * (1.0 - self.promotion_margin):
            selected = self.challenger
            reason = "challenger cleared the predefined historical promotion margin"
        else:
            selected = self.champion
            reason = "challenger did not clear the predefined promotion margin"
        return SelectionDecision(
            policy="champion_challenger",
            selected_models=(selected,),
            weights=(1.0,),
            reason=reason,
            scores=tuple(eligible),
        )


@dataclass(frozen=True)
class AutoSelectionPolicy:
    mode: str = "select"
    minimum_observations: int = 100
    incumbent: str | None = None
    switch_margin: float = 0.02

    def select(
        self, scores: list[HistoricalScore], forecast_origin: pd.Timestamp
    ) -> SelectionDecision:
        if self.mode not in {"select", "ensemble"}:
            raise ValueError("mode must be 'select' or 'ensemble'")
        eligible = sorted(
            _eligible(scores, forecast_origin, self.minimum_observations),
            key=lambda score: score.mae,
        )
        if not eligible:
            raise InsufficientEvidenceError(
                "no model has sufficient historical validation evidence"
            )
        if self.mode == "ensemble" and len(eligible) >= 2:
            selected = tuple(score.model_name for score in eligible)
            inverse = [1.0 / max(score.mae, 1e-12) for score in eligible]
            total = sum(inverse)
            weights = tuple(value / total for value in inverse)
            reason = "inverse-MAE weights derived from historical out-of-sample scores"
        else:
            best = eligible[0]
            incumbent_score = next(
                (score for score in eligible if score.model_name == self.incumbent), None
            )
            if (
                incumbent_score is not None
                and best.model_name != self.incumbent
                and best.mae >= incumbent_score.mae * (1.0 - self.switch_margin)
            ):
                best = incumbent_score
                reason = "incumbent retained to avoid a switch below the hysteresis margin"
            else:
                reason = "lowest eligible historical out-of-sample MAE"
            selected = (best.model_name,)
            weights = (1.0,)
        return SelectionDecision(
            policy=f"auto_{self.mode}",
            selected_models=selected,
            weights=weights,
            reason=reason,
            scores=tuple(eligible),
        )
