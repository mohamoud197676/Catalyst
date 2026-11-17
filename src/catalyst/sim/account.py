"""The six-month question, answered by simulation instead of hope.

Two parts:
  1. H1 account paths: 2,500 pounds, the event outcome mixture, realistic event counts and
     holding periods, several sizing rules. Probability of reaching 100,000 in six months,
     and of losing half.
  2. The requirement in reverse: for ANY strategy, how many independent trades and how much
     net edge per trade would be needed for the median path to reach 40x in six months.

Growth theory used in part 2: betting a fraction f of equity on trades with mean net return mu
and standard deviation sigma gives expected log growth per trade of about f*mu - f^2 sigma^2 / 2,
maximised at the Kelly fraction f* = mu / sigma^2 with growth mu^2 / (2 sigma^2).
"""
from __future__ import annotations

import math

import numpy as np

from .validate import COST, MIXTURE, _draw_outcomes, _mixture_mean

START, TARGET = 2_500.0, 100_000.0


def h1_paths(rng, true_net_excess: float, events_per_6m: float, size: float, max_concurrent: int,
             resolve_share_6m: float = 0.45, trials: int = 20_000) -> dict:
    """Simulate six months of the H1 book.

    Events arrive as a Poisson stream. Each position takes `size` of equity at entry. A position
    that resolves inside six months realises its outcome; one that does not is marked at a
    partial outcome (half the move so far), because most H1 resolutions take longer than six months.
    """
    base = _mixture_mean(rng)
    ends = np.empty(trials)
    for t in range(trials):
        equity, open_slots = START, max_concurrent
        n = rng.poisson(events_per_6m)
        for _ in range(n):
            if open_slots == 0:
                break
            raw, _ = _draw_outcomes(rng, 1)
            r = raw[0] - base + true_net_excess
            if rng.random() > resolve_share_6m:
                r *= 0.5
            stake = equity * size
            equity += stake * r
            open_slots -= 1 if rng.random() > resolve_share_6m else 0
        ends[t] = equity
    return {"median": float(np.median(ends)), "p_target": float(np.mean(ends >= TARGET)),
            "p_lose_half": float(np.mean(ends <= START / 2)), "p_10x": float(np.mean(ends >= 10 * START)),
            "p95": float(np.percentile(ends, 95))}


def required_edge(trades: int, sigma: float, kelly_share: float = 1.0, multiple: float = TARGET / START) -> float:
    """Net mean return per trade needed for the MEDIAN path to hit `multiple` after `trades` trades.

    At a fraction k of Kelly, growth per trade = (k - k^2/2) * mu^2 / sigma^2.
    """
    need = math.log(multiple) / trades
    return sigma * math.sqrt(need / (kelly_share - kelly_share ** 2 / 2))


def report(seed: int = 11) -> dict:
    rng = np.random.default_rng(seed)
    scenarios = []
    for label, edge, events in (("No edge", 0.0, 6), ("Realistic if H1 is real", 0.10, 6),
                                ("Optimistic", 0.20, 10), ("Exceptional", 0.30, 15)):
        for size, conc, rule in ((0.10, 10, "10% per position (frozen rule)"),
                                 (0.33, 3, "33% per position, 3 at a time"),
                                 (1.00, 1, "everything in one position")):
            scenarios.append({"scenario": label, "edge": edge, "events": events, "sizing": rule,
                              **h1_paths(rng, edge, events, size, conc, trials=8_000)})
    reverse = [{"trades_in_6m": n, "sigma": s,
                "edge_full_kelly": required_edge(n, s), "edge_half_kelly": required_edge(n, s, 0.5)}
               for n in (10, 50, 250, 1000) for s in (0.30,)]
    return {"h1": scenarios, "reverse": reverse, "ln_target": math.log(TARGET / START)}


def markov_ceiling(expected_six_month_return: float, start: float = START, target: float = TARGET) -> float:
    """Upper bound on P(equity >= target at six months) for ANY strategy, however clever.

    Equity cannot go below zero, so Markov's inequality gives P(X >= target) <= E[X] / target.
    A strategy with zero expected return has at most a 2.5% chance of 2,500 -> 100,000. No sizing
    rule, option structure or AI can beat this; only a higher expected return raises the ceiling.
    """
    return min(1.0, start * (1 + expected_six_month_return) / target)
