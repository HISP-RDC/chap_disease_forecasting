"""Paired comparison of two CHAP backtests run on the same zones and splits.

Usage (on the CHAP host; standard library only):

    export CHAP_API_TOKEN=...            # only if CHAP sets one
    python3 compare_backtests.py 71 72   # backtest A (candidate), backtest B (reference)

Every forecast of A is matched with the forecast of B for the same zone, week
and forecast origin, and scored against the observed cases from CHAP:

- CRPS from the predictive samples and absolute error of the median;
- the share of matched forecasts, and of zones, where A scores better;
- a two-sided sign test over zones (does A win in more zones than chance?);
- a 95 % confidence interval of the CRPS and MAE ratios A/B, by bootstrap over
  zones (forecasts of one zone are correlated, so zones are resampled whole);
- CRPS by horizon.

CHAP_URL defaults to http://localhost:8000.
"""

from __future__ import annotations

import datetime
import json
import math
import os
import random
import re
import statistics
import sys
import urllib.request
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

_PERIOD = re.compile(r"^(\d{4})-?W(\d{1,2})$")
Key = tuple[str, str, str]  # (zone, period, origin)


def _get(url: str, token: str | None) -> Any:
    if not url.startswith(("http://", "https://")):
        raise ValueError("CHAP_URL must be an http(s) URL")
    request = urllib.request.Request(url)  # noqa: S310 - scheme checked above
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    with urllib.request.urlopen(request, timeout=300) as response:  # noqa: S310
        return json.load(response)


def _field(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row:
            return row[name]
    raise KeyError(names[0])


def week_index(period: str) -> int:
    match = _PERIOD.match(str(period))
    if not match:
        raise ValueError(f"not a weekly period: {period!r}")
    # ISO years have 52 or 53 weeks: count weeks from the real Monday of each week.
    monday = datetime.date.fromisocalendar(int(match.group(1)), int(match.group(2)), 1)
    return monday.toordinal() // 7


def crps(samples: Sequence[float], observed: float) -> float:
    """Sample CRPS: mean |X - y| - 0.5 mean |X - X'|."""

    ordered = sorted(samples)
    n = len(ordered)
    spread = sum(value * (2 * (index + 1) - n - 1) for index, value in enumerate(ordered))
    return sum(abs(value - observed) for value in ordered) / n - spread / (n * n)


def median(samples: Sequence[float]) -> float:
    return float(statistics.median(samples))


def forecasts_by_key(backtest: Mapping[str, Any]) -> dict[Key, list[float]]:
    rows: dict[Key, list[float]] = {}
    for row in backtest["forecasts"]:
        key = (
            str(_field(row, "orgUnit", "org_unit")),
            str(row["period"]),
            str(_field(row, "lastSeenPeriod", "last_seen_period")),
        )
        values = [float(value) for value in row["values"]]
        if values:
            rows[key] = values
    return rows


def observed_by_key(actual: Mapping[str, Any]) -> dict[tuple[str, str], float]:
    return {
        (str(item["ou"]), str(item["pe"])): float(item["value"])
        for item in actual["data"]
        if item.get("value") is not None
    }


def _ratio_interval(
    per_zone: Mapping[str, tuple[float, float, int]], resamples: int, seed: int
) -> tuple[float, float]:
    zones = list(per_zone)
    rng = random.Random(seed)  # noqa: S311 - statistical resampling, not security
    ratios = []
    for _ in range(resamples):
        drawn = [per_zone[rng.choice(zones)] for _ in zones]
        total_a = sum(a for a, _, _ in drawn)
        total_b = sum(b for _, b, _ in drawn)
        if total_b > 0:
            ratios.append(total_a / total_b)
    ratios.sort()
    return ratios[int(0.025 * len(ratios))], ratios[int(0.975 * len(ratios)) - 1]


def sign_test(wins: int, losses: int) -> float:
    """Two-sided binomial sign test, ties excluded."""

    n = wins + losses
    if n == 0:
        return 1.0
    extreme = min(wins, losses)
    tail = sum(math.comb(n, k) for k in range(extreme + 1)) / 2**n
    return min(1.0, 2 * tail)


def compare(
    forecasts_a: Mapping[Key, Sequence[float]],
    forecasts_b: Mapping[Key, Sequence[float]],
    observed: Mapping[tuple[str, str], float],
    resamples: int = 2000,
    seed: int = 20261026,
) -> dict[str, Any]:
    keys = sorted(
        key for key in set(forecasts_a) & set(forecasts_b) if (key[0], key[1]) in observed
    )
    if not keys:
        raise ValueError("no forecast is shared by both backtests and observed")
    crps_zone: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    ae_zone: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    by_horizon: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    forecast_wins = 0
    for zone, period, origin in keys:
        truth = observed[(zone, period)]
        score_a = crps(forecasts_a[(zone, period, origin)], truth)
        score_b = crps(forecasts_b[(zone, period, origin)], truth)
        error_a = abs(median(forecasts_a[(zone, period, origin)]) - truth)
        error_b = abs(median(forecasts_b[(zone, period, origin)]) - truth)
        forecast_wins += score_a < score_b
        for table, a, b in ((crps_zone, score_a, score_b), (ae_zone, error_a, error_b)):
            table[zone][0] += a
            table[zone][1] += b
            table[zone][2] += 1
        horizon = week_index(period) - week_index(origin)
        by_horizon[horizon][0] += score_a
        by_horizon[horizon][1] += score_b
        by_horizon[horizon][2] += 1
    zone_wins = sum(a < b for a, b, _ in crps_zone.values())
    zone_losses = sum(a > b for a, b, _ in crps_zone.values())
    n = len(keys)

    def mean(table: Mapping[str, Sequence[float]], column: int) -> float:
        return sum(row[column] for row in table.values()) / n

    crps_zone_t = {zone: (a, b, int(c)) for zone, (a, b, c) in crps_zone.items()}
    ae_zone_t = {zone: (a, b, int(c)) for zone, (a, b, c) in ae_zone.items()}
    return {
        "matched_forecasts": n,
        "zones": len(crps_zone),
        "crps": (mean(crps_zone, 0), mean(crps_zone, 1)),
        "mae_median": (mean(ae_zone, 0), mean(ae_zone, 1)),
        "crps_ratio_ci95": _ratio_interval(crps_zone_t, resamples, seed),
        "mae_ratio_ci95": _ratio_interval(ae_zone_t, resamples, seed),
        "forecasts_won": forecast_wins / n,
        "zones_won": (zone_wins, zone_losses),
        "sign_test_p": sign_test(zone_wins, zone_losses),
        "crps_by_horizon": {
            h: (a / c, b / c, int(c)) for h, (a, b, c) in sorted(by_horizon.items())
        },
    }


def report(result: Mapping[str, Any], names: Iterable[str]) -> str:
    a_name, b_name = names
    crps_a, crps_b = result["crps"]
    mae_a, mae_b = result["mae_median"]
    low, high = result["crps_ratio_ci95"]
    mae_low, mae_high = result["mae_ratio_ci95"]
    wins, losses = result["zones_won"]
    lines = [
        f"A = {a_name}   B = {b_name}",
        f"matched forecasts: {result['matched_forecasts']} in {result['zones']} zones",
        f"CRPS          A {crps_a:.3f}   B {crps_b:.3f}   ratio A/B {crps_a / crps_b:.3f}"
        f"   95% CI [{low:.3f}, {high:.3f}]",
        f"MAE (median)  A {mae_a:.3f}   B {mae_b:.3f}   ratio A/B {mae_a / mae_b:.3f}"
        f"   95% CI [{mae_low:.3f}, {mae_high:.3f}]",
        f"A has the lower CRPS in {result['forecasts_won']:.1%} of matched forecasts",
        f"A wins in {wins} zones, B in {losses} (sign test p = {result['sign_test_p']:.2g})",
        "CRPS by horizon (weeks ahead):",
    ]
    lines += [
        f"  {h}: A {a:.3f}   B {b:.3f}   ({c} forecasts)"
        for h, (a, b, c) in result["crps_by_horizon"].items()
    ]
    lines.append("A ratio below 1 whose whole interval stays below 1 means A is better.")
    return "\n".join(lines)


def main(argv: Sequence[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    base = os.environ.get("CHAP_URL", "http://localhost:8000").rstrip("/")
    token = os.environ.get("CHAP_API_TOKEN") or None
    first, second = argv
    backtest_a = _get(f"{base}/v1/crud/backtests/{first}/full", token)
    backtest_b = _get(f"{base}/v1/crud/backtests/{second}/full", token)
    observed = observed_by_key(_get(f"{base}/v1/analytics/actual-cases/{first}", token))
    observed_b = observed_by_key(_get(f"{base}/v1/analytics/actual-cases/{second}", token))
    shared = set(observed) & set(observed_b)
    if any(observed[key] != observed_b[key] for key in shared):
        print("warning: the two datasets disagree on some observed cases", file=sys.stderr)
    result = compare(forecasts_by_key(backtest_a), forecasts_by_key(backtest_b), observed)
    names = (
        str(_field(backtest_a, "modelId", "model_id")),
        str(_field(backtest_b, "modelId", "model_id")),
    )
    print(report(result, names))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
