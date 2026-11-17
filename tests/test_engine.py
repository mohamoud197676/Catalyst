"""Corporate Move Engine, event verdict and the six-month arithmetic."""
import math

import numpy as np
import pytest

from catalyst.backtest.events import Event, clustered_t, verdict
from catalyst.engine import model as m
from catalyst.sim import account


def state(**kw):
    base = dict(as_of="2024-11-08", price=1.50, shares=20e6, net_cash=60e6, quarterly_burn=5e6,
                activist_13d=True)
    return m.CompanyState(**{**base, **kw})


def synthetic_panel(rng, n=600):
    """Companies whose true next move depends on discount and trigger, with known odds."""
    rows = []
    for i in range(n):
        s = state(price=float(rng.uniform(0.5, 4.0)), activist_13d=bool(rng.random() < 0.5),
                  quarterly_burn=float(rng.uniform(1e6, 10e6)))
        x = m.features(s)
        logits = np.array([2.5 * x[0] + 1.5 * x[3] - 1, -0.5, 0.3 - x[0], 0.5, -1.5 - x[1]])
        p = np.exp(logits - logits.max()); p /= p.sum()
        move = m.MOVES[rng.choice(5, p=p)]
        year = 2012 + i % 10
        rows.append({"state": s, "move": move, "outcome_known": f"{year + 1}-06-30", "true_p": p})
    return rows


def test_state_arithmetic():
    s = state()
    assert s.market_value == 30e6 and s.runway_quarters == 12
    assert s.adjusted_net_cash() == 50e6 and s.discount() == pytest.approx(0.4)
    assert s.trigger() and s.supply_overhang() == []
    assert state(net_cash=5e6).discount() == -math.inf


def test_walk_forward_fit_never_uses_unresolved_outcomes():
    rng = np.random.default_rng(1)
    panel = synthetic_panel(rng)
    assert m.fit_asof(panel, "2012-12-31") is None                       # nothing resolved yet
    early = m.fit_asof(panel, "2015-07-01")
    late = m.fit_asof(panel, "2023-07-01")
    assert early.n_train == sum(r["outcome_known"] <= "2015-07-01" for r in panel) < late.n_train == len(panel)


def test_move_model_recovers_true_probabilities_better_than_base_rates():
    rng = np.random.default_rng(2)
    train, test = synthetic_panel(rng, 2000), synthetic_panel(rng, 500)
    model = m.MoveModel().fit(np.vstack([m.features(r["state"]) for r in train]), [r["move"] for r in train])
    P = model.predict(np.vstack([m.features(r["state"]) for r in test]))
    y = [r["move"] for r in test]
    base = np.tile(np.bincount([m.MOVES.index(r["move"]) for r in train], minlength=5) / len(train), (len(test), 1))
    assert m.brier(P, y) < m.brier(base, y)                          # it learned something real
    true = np.vstack([r["true_p"] for r in test])
    assert np.abs(P - true).mean() < 0.06                            # and close to the truth


def test_payoff_book_learns_only_resolved_history():
    hist = [{"move": m.REVERSE_MERGER, "multiple": 0.4, "resolved": "2018-01-01"}] * 20 + \
           [{"move": m.REVERSE_MERGER, "multiple": 2.0, "resolved": "2024-01-01"}] * 30
    assert m.PayoffBook().learn(hist, "2019-01-01").reverse_merger == 0.4
    assert m.PayoffBook().learn(hist, "2017-01-01").reverse_merger == 0.70   # prior kept: too few samples
    s = state()
    # cash return: (60m - 3 x 5m) x 0.8 / 20m shares = 1.80 per share, against a 1.50 price
    assert m.PayoffBook().cash_return_multiple(s) == pytest.approx(1.2)


def test_two_ply_search_and_decision_gates():
    rng = np.random.default_rng(3)
    panel = synthetic_panel(rng, 1500)
    model = m.fit_asof(panel, "2030-01-01")
    book = m.PayoffBook()
    ev = m.evaluate(state(price=1.0), model, book)
    assert sum(ev.move_probs.values()) == pytest.approx(1.0)
    assert any(len(path) == 2 for _, path, _ in ev.line)           # looked past a dilution
    assert 0 <= ev.p_loss_over_half <= 1
    pol = m.Policy()
    no = m.decide(state(activist_13d=False, atm_open=True), model, book, pol, daily_value=500_000)
    assert not no.act and any("trigger" in r for r in no.reasons) and any("overhang" in r for r in no.reasons)
    assert not m.decide(state(), None, book, pol, 500_000).act        # untrained model never trades
    yes = m.decide(state(price=1.10), model, book, pol, 500_000)          # deep discount, triggered
    assert yes.act and yes.size_fraction == pol.max_position_fraction     # Kelly capped by the 10% rule
    pricey = m.decide(state(price=2.00), model, book, pol, 500_000)       # still below cash, but not by enough
    assert not pricey.act and "below hurdle" in pricey.reasons[0]
    assert not m.decide(state(price=0.80), model, book, pol, 500_000).act    # below 1 dollar


def ev(year, q, gross, bench=0.0, cost=0.03, tradeable=True):
    return Event("X", f"{year}-{3 * q - 1:02d}-15", gross, cost, bench, tradeable=tradeable)


def test_verdict_applies_every_frozen_check():
    good = [ev(2020 + i % 6, 1 + i % 4, 0.45 if i % 5 else -0.2) for i in range(80)]
    dev = [ev(2012, 1, 0.3)] * 90
    v = verdict(good, dev)
    assert v.passed, v.failed()
    lucky = [ev(2020 + i % 6, 1 + i % 4, -0.10) for i in range(76)] + [ev(2021, 2, 9.0)] * 4
    lv = verdict(lucky, dev)
    assert not lv.passed and "survives_best_5pct_removed" in lv.failed() and "median_positive" in lv.failed()
    few = verdict(good[:40], dev)
    assert not few.passed and "enough_holdout_events" in few.failed()
    illiquid = verdict([ev(2020 + i % 6, 1 + i % 4, 0.45, tradeable=i < 50) for i in range(80)], dev)
    assert illiquid.stats["untradeable"] == 30 and not illiquid.checks["enough_holdout_events"]


def test_clustered_t_shrinks_when_one_quarter_drives_the_result():
    x = np.array([0.5] * 10 + [0.0] * 50)
    spread = [f"{2020 + i % 6}Q{1 + i % 4}" for i in range(60)]
    lumped = ["2021Q1"] * 10 + [f"{2020 + i % 6}Q{1 + i % 4}" for i in range(50)]
    assert clustered_t(x, lumped) < clustered_t(x, spread)


def test_six_month_arithmetic():
    assert account.markov_ceiling(0.0) == 0.025          # zero edge: at most 2.5%, whatever the method
    assert account.markov_ceiling(1.0) == 0.05           # doubling expected: still at most 5%
    e = account.required_edge(250, 0.30)
    n_growth = 250 * (e ** 2 / (2 * 0.30 ** 2))
    assert n_growth == pytest.approx(math.log(40))
