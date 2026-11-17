import math

import pytest

from catalyst.sim import frontier as fr


def test_binomial_tail_matches_direct_count():
    assert fr.binom_tail(10, 0, 0.3) == 1.0
    assert fr.binom_tail(10, 11, 0.3) == 0.0
    assert fr.binom_tail(4, 4, 0.5) == pytest.approx(1 / 16)


def test_no_edge_has_no_kelly_bet_and_target_needs_luck():
    assert fr.kelly_fraction(0.5, 1.0, cost=0.0) == 0.0          # fair coin: do not bet
    o = fr.evaluate(0.5, 1.0, 0.5, 20)
    assert o.edge_per_trade == 0.0 and o.p_target < 0.025         # inside the Markov ceiling


def test_known_case_and_monotonicity():
    # 60% wins, 2:1 payoff, risk 20%, 50 trades: plenty of edge, target reachable but not certain
    o = fr.evaluate(0.6, 2.0, 0.2, 50)
    assert 0 < o.p_target < 1 and o.edge_per_trade == pytest.approx(0.8)
    assert fr.evaluate(0.6, 2.0, 0.2, 100).p_target > o.p_target    # more trades help
    assert fr.evaluate(0.6, 2.0, 0.2, 50, cost=0.05).p_target < o.p_target   # costs hurt
    k = fr.kelly_fraction(0.6, 2.0)
    assert k == pytest.approx(0.4, abs=0.002)                       # Kelly: p - (1-p)/R = 0.4
