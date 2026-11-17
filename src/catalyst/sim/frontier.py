"""Target frontier: what win rate, win:loss ratio, exposure and trade count a target requires.

A trade risks fraction f of equity. It wins with probability p and returns f * R (R is the
win:loss ratio, average win over average loss), or loses f. Costs c (as a fraction of the
position) come off every trade. With N trades in the window and k wins, equity is

    E = E0 * (1 + f*R - c*f)^k * (1 - f - c*f)^(N - k)

so the chance of reaching the target is an exact binomial tail. No simulation noise, and no
way to argue with it: it is the arithmetic every strategy, human or AI, is bound by.

This is the tool the system uses to answer "what are our chances?" once a strategy's win rate
and win:loss ratio have been MEASURED out of sample. Plug in measured values, not hoped-for ones.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

START, TARGET = 2_500.0, 100_000.0


def _log_binom_pmf(n: int, k: int, p: float) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + k * math.log(p) + (n - k) * math.log1p(-p)


def binom_tail(n: int, k_min: int, p: float) -> float:
    if k_min <= 0:
        return 1.0
    if k_min > n:
        return 0.0
    return min(1.0, sum(math.exp(_log_binom_pmf(n, k, p)) for k in range(k_min, n + 1)))


@dataclass(frozen=True)
class Outcome:
    p_target: float          # chance of ending at or above the target
    p_lose_half: float       # chance of ending at or below half the start
    median_end: float
    wins_needed: int
    edge_per_trade: float    # expected return per unit risked, after costs
    growth_per_trade: float  # expected log growth per trade


def evaluate(p: float, R: float, f: float, N: int, cost: float = 0.0,
             start: float = START, target: float = TARGET) -> Outcome:
    up, down = 1 + f * R - cost * f, 1 - f - cost * f
    if down <= 0:
        return Outcome(0.0, 1.0, 0.0, N + 1, p * R - (1 - p) - cost, -math.inf)
    lu, ld = math.log(up), math.log(down)
    need = math.log(target / start)
    # smallest k with k*lu + (N-k)*ld >= need
    k_t = math.ceil((need - N * ld) / (lu - ld)) if lu > ld else N + 1
    k_h = math.floor((math.log(0.5) - N * ld) / (lu - ld)) if lu > ld else -1
    p_half = 1 - binom_tail(N, k_h + 1, p) if k_h >= 0 else 0.0
    k_med = round(N * p)
    median = start * math.exp(k_med * lu + (N - k_med) * ld)
    return Outcome(binom_tail(N, max(k_t, 0), p), p_half, median, max(k_t, 0),
                   p * R - (1 - p) - cost, p * lu + (1 - p) * ld)


def kelly_fraction(p: float, R: float, cost: float = 0.0) -> float:
    """Growth-optimal fraction of equity to risk per trade (0 when there is no edge)."""
    grid = [i / 1000 for i in range(1, 1000)]
    best = max(grid, key=lambda f: evaluate(p, R, f, 1, cost).growth_per_trade)
    return best if evaluate(p, R, best, 1, cost).growth_per_trade > 0 else 0.0


def best_chance(p: float, R: float, N: int, cost: float = 0.0) -> tuple[float, Outcome]:
    """The exposure that MAXIMISES the chance of hitting the target (not the growth-optimal one).

    This is the most aggressive sizing that could be justified by the target alone. It is
    reported so the trade-off is visible: it buys target probability with ruin probability.
    """
    grid = [i / 100 for i in range(1, 100)]
    f = max(grid, key=lambda x: evaluate(p, R, x, N, cost).p_target)
    return f, evaluate(p, R, f, N, cost)


def frontier_table(win_rates=(0.40, 0.50, 0.60), ratios=(1.0, 2.0, 3.0), trades=(20, 50, 100, 250),
                   cost: float = 0.02) -> list[dict]:
    rows = []
    for p in win_rates:
        for R in ratios:
            k = kelly_fraction(p, R, cost)
            for N in trades:
                safe = evaluate(p, R, k / 2, N, cost) if k > 0 else None
                f_max, agg = best_chance(p, R, N, cost)
                rows.append({"win_rate": p, "win_loss": R, "trades": N, "edge": p * R - (1 - p) - cost,
                             "half_kelly_f": k / 2, "half_kelly_p_target": safe.p_target if safe else 0.0,
                             "half_kelly_p_lose_half": safe.p_lose_half if safe else None,
                             "half_kelly_median": safe.median_end if safe else START,
                             "max_chance_f": f_max, "max_chance_p_target": agg.p_target,
                             "max_chance_p_lose_half": agg.p_lose_half})
    return rows
