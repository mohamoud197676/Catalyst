"""Event backtester and the pre-registered verdict for H1.

Input is a list of realised events, one per company entry, built point-in-time by the
pipeline. This module only scores them; it cannot see prices, so it cannot peek.

The verdict applies the frozen pass marks exactly. It does not report a 'best' variant.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Event:
    company: str
    entry_date: str                  # YYYY-MM-DD
    gross_return: float              # entry price to exit value incl. distributions, before costs
    cost: float                      # round-trip cost as a fraction of entry
    benchmark_return: float          # matched no-trigger below-cash group, same window
    index_return: float = 0.0        # small-cap index, same window
    exit_reason: str = "TIME"        # TIME | RESOLVED | NET_CASH_BREACH | DELISTED
    tradeable: bool = True           # False when the order would exceed 1% of volume

    @property
    def year(self) -> int:
        return int(self.entry_date[:4])

    @property
    def quarter(self) -> str:
        return f"{self.entry_date[:4]}Q{(int(self.entry_date[5:7]) - 1) // 3 + 1}"

    def net(self, cost_mult: float = 1.0) -> float:
        return self.gross_return - cost_mult * self.cost


@dataclass(frozen=True)
class PassMarks:
    min_holdout_events: int = 60
    min_total_events: int = 150
    t_stat: float = 3.0
    best_trim: float = 0.05
    min_positive_years: int = 4
    worst_year_floor: float = -0.25
    max_share_big_loss: float = 0.15
    big_loss: float = -0.50


def clustered_t(x: np.ndarray, clusters: list[str]) -> float:
    """t-statistic of the mean with standard errors clustered by calendar quarter.

    Events in the same quarter share one market; treating them as independent would
    overstate certainty, which is how most event backtests reach significance.
    """
    n = len(x)
    if n < 3:
        return 0.0
    m = x.mean()
    groups: dict[str, float] = {}
    for v, g in zip(x, clusters):
        groups[g] = groups.get(g, 0.0) + (v - m)
    G = len(groups)
    if G < 2:
        return 0.0
    var = sum(s * s for s in groups.values()) / n ** 2 * G / (G - 1)
    return float(m / math.sqrt(var)) if var > 0 else 0.0


@dataclass
class Verdict:
    passed: bool
    checks: dict = field(default_factory=dict)
    stats: dict = field(default_factory=dict)

    def failed(self) -> list[str]:
        return [k for k, v in self.checks.items() if not v]


def verdict(holdout: list[Event], development: list[Event], marks: PassMarks = PassMarks()) -> Verdict:
    tradeable = [e for e in holdout if e.tradeable]
    n_hold, n_total = len(tradeable), len(tradeable) + sum(e.tradeable for e in development)
    v = Verdict(False)
    v.stats.update(n_holdout=n_hold, n_total=n_total, untradeable=len(holdout) - n_hold)
    v.checks["enough_holdout_events"] = n_hold >= marks.min_holdout_events
    v.checks["enough_total_events"] = n_total >= marks.min_total_events
    if n_hold < 3:
        return v
    net = np.array([e.net() for e in tradeable])
    excess = np.array([e.net() - e.benchmark_return for e in tradeable])
    t = clustered_t(excess, [e.quarter for e in tradeable])
    trimmed = np.sort(net)[: max(1, int(math.floor(len(net) * (1 - marks.best_trim))))]
    by_year: dict[int, list[float]] = {}
    for e in tradeable:
        by_year.setdefault(e.year, []).append(e.net())
    year_means = {y: float(np.mean(r)) for y, r in sorted(by_year.items())}
    v.stats.update(median=float(np.median(net)), mean=float(net.mean()), mean_excess=float(excess.mean()),
                   t_clustered=t, mean_trimmed=float(trimmed.mean()),
                   mean_double_cost=float(np.mean([e.net(2.0) for e in tradeable])),
                   year_means=year_means, share_big_loss=float(np.mean(net < marks.big_loss)),
                   mean_vs_index=float(np.mean([e.net() - e.index_return for e in tradeable])))
    v.checks["median_positive"] = v.stats["median"] > 0
    v.checks["beats_no_trigger_t3"] = v.stats["mean_excess"] > 0 and t > marks.t_stat
    v.checks["survives_best_5pct_removed"] = v.stats["mean_trimmed"] > 0
    v.checks["survives_double_costs"] = v.stats["mean_double_cost"] > 0
    v.checks["positive_years"] = sum(m > 0 for m in year_means.values()) >= marks.min_positive_years
    v.checks["worst_year"] = min(year_means.values()) >= marks.worst_year_floor
    v.checks["big_losses_rare"] = v.stats["share_big_loss"] < marks.max_share_big_loss
    v.checks["beats_index"] = v.stats["mean_vs_index"] > 0
    v.checks["beats_cash"] = v.stats["mean"] > 0
    v.passed = all(v.checks.values())
    return v
