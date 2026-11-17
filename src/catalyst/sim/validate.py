"""Test the test: before any real data, measure how often the frozen H1 verdict is fooled.

A synthetic world is generated where the true edge is known. Three questions:
  1. False positives: with NO edge after costs, how often does the verdict say PASS?
  2. Power: with a real edge of a given size, how often does it say PASS?
  3. Damage from common mistakes: dropping delisted companies, and ignoring that events in
     the same quarter share one market. How much do they inflate a no-edge result?

The synthetic outcome mixture is an assumption shaped like small-cap event outcomes
(some cash returns, some reverse mergers, some delistings, many drifts). It proves
nothing about real markets; it measures the instrument, not the world.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..backtest.events import Event, PassMarks, verdict

# Outcome mixture relative to the matched benchmark, before costs: (share, mean, sd, label)
MIXTURE = [
    (0.30, 0.30, 0.25, "CASH_RETURN"),
    (0.15, -0.40, 0.25, "REVERSE_MERGER"),
    (0.08, -0.85, 0.08, "DELIST"),
    (0.47, -0.05, 0.45, "STATUS_QUO"),
]
QUARTER_SHOCK_SD = 0.15        # shared by every event entered in the same quarter
COST = 0.03


def _draw_outcomes(rng, n: int) -> tuple[np.ndarray, list[str]]:
    shares = np.array([m[0] for m in MIXTURE])
    idx = rng.choice(len(MIXTURE), size=n, p=shares / shares.sum())
    vals = np.array([rng.normal(MIXTURE[i][1], MIXTURE[i][2]) for i in idx])
    return np.maximum(vals, -0.99), [MIXTURE[i][3] for i in idx]


def _mixture_mean(rng) -> float:
    v, _ = _draw_outcomes(rng, 200_000)
    return float(v.mean())


def make_events(rng, n: int, years: range, true_net_excess: float, base_mean: float,
                cluster: bool = True) -> list[Event]:
    """Events whose expected excess return AFTER costs equals true_net_excess."""
    raw, labels = _draw_outcomes(rng, n)
    excess = raw - base_mean + true_net_excess + COST
    yrs = rng.choice(list(years), size=n)
    qs = rng.integers(1, 5, size=n)
    shocks = {}
    out = []
    for i in range(n):
        key = (yrs[i], qs[i])
        if key not in shocks:
            shocks[key] = rng.normal(0, QUARTER_SHOCK_SD) if cluster else 0.0
        bench = shocks[key] + rng.normal(0, 0.03)
        gross = max(bench + excess[i], -0.99)
        out.append(Event(company=f"C{i}", entry_date=f"{yrs[i]}-{3 * qs[i] - 1:02d}-15", gross_return=gross,
                         cost=COST, benchmark_return=bench, index_return=shocks[key],
                         exit_reason="DELISTED" if labels[i] == "DELIST" else "TIME"))
    return out


@dataclass
class Rate:
    label: str
    pass_rate: float
    trials: int


def run(trials: int = 1000, seed: int = 7, n_hold: int = 60, n_dev: int = 90) -> dict:
    rng = np.random.default_rng(seed)
    base = _mixture_mean(rng)
    hold_years, dev_years = range(2020, 2026), range(2010, 2020)
    results: dict[str, list] = {"power": [], "mistakes": []}

    def rate(edge: float, survivorship: bool = False, naive_t: bool = False, nh: int = n_hold) -> float:
        passes = 0
        for _ in range(trials):
            hold = make_events(rng, nh, hold_years, edge, base)
            dev = make_events(rng, n_dev, dev_years, edge, base)
            if survivorship:
                hold = [e for e in hold if e.exit_reason != "DELISTED"]
                dev = [e for e in dev if e.exit_reason != "DELISTED"]
            if naive_t:   # every event its own cluster: the independence mistake
                hold = [Event(**{**e.__dict__, "entry_date": f"{2020 + i % 6}-{(i % 12) + 1:02d}-{(i // 12) % 28 + 1:02d}"})
                        for i, e in enumerate(hold)]
            passes += verdict(hold, dev).passed
        return passes / trials

    for edge in (0.0, 0.05, 0.10, 0.20, 0.30, 0.40):
        results["power"].append({"true_net_excess": edge, "pass_rate_60": rate(edge),
                                 "pass_rate_120": rate(edge, nh=120)})
    results["mistakes"] = [
        {"case": "honest test, no edge", "pass_rate": rate(0.0)},
        {"case": "delisted companies dropped, no edge", "pass_rate": rate(0.0, survivorship=True)},
        {"case": "delisted dropped, true edge -10%", "pass_rate": rate(-0.10, survivorship=True)},
    ]
    # how much does dropping dead companies flatter the mean?
    hold = make_events(rng, 20_000, hold_years, 0.0, base)
    alive = [e.net() - e.benchmark_return for e in hold if e.exit_reason != "DELISTED"]
    allx = [e.net() - e.benchmark_return for e in hold]
    results["survivorship_inflation"] = float(np.mean(alive) - np.mean(allx))
    results["naive_vs_clustered"] = _t_inflation(rng, base)
    results["base_mixture"] = {"mean": base, "median_net_at_zero_edge": float(np.median([e.net() for e in hold]))}
    return results


def _t_inflation(rng, base: float, trials: int = 2000) -> dict:
    """Share of no-edge samples with |t| > 3 under naive vs clustered standard errors."""
    from ..backtest.events import clustered_t
    naive = clustered_hits = 0
    for _ in range(trials):
        ev = make_events(rng, 60, range(2020, 2026), 0.0, base)
        x = np.array([e.net() - e.benchmark_return for e in ev])
        t_naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
        naive += abs(t_naive) > 3
        clustered_hits += abs(clustered_t(x, [e.quarter for e in ev])) > 3
    return {"naive_t_over_3": naive / trials, "clustered_t_over_3": clustered_hits / trials}
