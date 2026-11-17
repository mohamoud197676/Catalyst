"""The Corporate Move Engine.

Most trading systems try to predict the next price from past prices. This engine predicts
something narrower and more knowable: the next MOVE a cash-constrained company is forced to
make, and what each move pays a shareholder.

A small company that is burning cash has a short list of legal moves, and each is preceded
by filings: raise money (needs a shelf and capacity), reverse split (needs a vote, has a Nasdaq
clock), sell itself, merge into a private company, wind down and return cash, or keep burning
until it cannot. Like a chess engine, it lists those moves, estimates how likely each is from
the position, values each outcome, looks two moves ahead, and plays only when the expected
value clears costs with a margin for being wrong.

Three things are LEARNED from history (point-in-time, walk-forward):
  1. move probabilities given the position (a multinomial hazard model)
  2. what each move has historically paid, per share, relative to the entry price
  3. how much of reported net cash survives to reach shareholders (the recovery rate)

Everything else is explicit and auditable. There is no neural network and no price pattern.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

# --------------------------------------------------------------------------- moves
CASH_RETURN = "CASH_RETURN"            # liquidation, tender or cash sale
REVERSE_MERGER = "REVERSE_MERGER"      # merges into a private company; holders diluted
DILUTIVE_RAISE = "DILUTIVE_RAISE"      # offering, ATM, PIPE, warrants
STATUS_QUO = "STATUS_QUO"              # nothing decisive; cash keeps burning
DELIST = "DELIST"                      # delisting, bankruptcy or deregistration
MOVES = (CASH_RETURN, REVERSE_MERGER, DILUTIVE_RAISE, STATUS_QUO, DELIST)
TERMINAL = {CASH_RETURN, REVERSE_MERGER, DELIST}

# Which filings reveal which move. Used to label history.
MOVE_EVIDENCE = {
    CASH_RETURN: "plan of liquidation (8-K, DEF 14A), SC TO-T, merger agreement with cash consideration",
    REVERSE_MERGER: "merger agreement with share consideration where target holders own the majority (8-K 1.01, S-4)",
    DILUTIVE_RAISE: "424B prospectus, ATM sales in 10-Q, 8-K 3.02, PIPE securities purchase agreement",
    DELIST: "Form 25, Form 15, 8-K 1.03, 8-K 3.01 followed by suspension",
}


# --------------------------------------------------------------------------- state
@dataclass(frozen=True)
class CompanyState:
    """Everything the engine may know about one company on one date. All point-in-time."""
    as_of: str
    price: float
    shares: float
    net_cash: float                 # cash + short-term investments - debt
    quarterly_burn: float           # >= 0
    shelf_capacity: float = math.inf    # dollars still sellable under a shelf (baby-shelf limit applied)
    atm_open: bool = False
    pending_resale: bool = False
    reset_terms: bool = False
    reverse_split_12m: bool = False
    below_one_dollar_days: int = 0
    activist_13d: bool = False
    strategic_review: bool = False
    liquidation_plan: bool = False
    tender_offer: bool = False
    insider_buying_90d: bool = False

    @property
    def market_value(self) -> float:
        return self.price * self.shares

    @property
    def runway_quarters(self) -> float:
        return math.inf if self.quarterly_burn <= 0 else self.net_cash / self.quarterly_burn

    def adjusted_net_cash(self, burn_quarters: float = 2.0) -> float:
        return self.net_cash - burn_quarters * self.quarterly_burn

    def discount(self, burn_quarters: float = 2.0) -> float:
        adj = self.adjusted_net_cash(burn_quarters)
        return -math.inf if adj <= 0 else 1.0 - self.market_value / adj

    def trigger(self) -> bool:
        return self.activist_13d or self.strategic_review or self.liquidation_plan or self.tender_offer

    def supply_overhang(self) -> list[str]:
        out = []
        if self.atm_open:
            out.append("open at-the-market programme")
        if self.pending_resale:
            out.append("resale registration pending")
        if self.reset_terms:
            out.append("price-reset financing")
        if self.reverse_split_12m:
            out.append("reverse split in last 12 months")
        return out


FEATURE_NAMES = ("discount", "log_runway", "log_mcap", "trigger", "activist", "review", "liquidation",
                 "tender", "insider_buy", "overhang", "below_dollar", "has_capacity")


def features(s: CompanyState) -> np.ndarray:
    d = s.discount()
    return np.array([
        max(min(d if math.isfinite(d) else -1.0, 1.0), -1.0),
        math.log(min(max(s.runway_quarters, 0.25), 40.0)),
        math.log(max(s.market_value, 1e5)),
        float(s.trigger()), float(s.activist_13d), float(s.strategic_review), float(s.liquidation_plan),
        float(s.tender_offer), float(s.insider_buying_90d), float(bool(s.supply_overhang())),
        min(s.below_one_dollar_days, 180) / 180.0, float(s.shelf_capacity > 0.1 * s.market_value),
    ])


# --------------------------------------------------------------------- hazard model
class MoveModel:
    """Multinomial logistic model: P(first decisive move within the horizon | position).

    Deliberately simple. With a few hundred labelled histories, anything more flexible
    memorises noise. L2 penalty, standardised inputs, plain gradient descent, deterministic.
    """

    def __init__(self, l2: float = 1.0, iters: int = 3000, lr: float = 0.1) -> None:
        self.l2, self.iters, self.lr = l2, iters, lr
        self.mu = self.sd = self.W = None
        self.n_train = 0

    def fit(self, X: np.ndarray, y: list[str]) -> "MoveModel":
        Y = np.zeros((len(y), len(MOVES)))
        for i, m in enumerate(y):
            Y[i, MOVES.index(m)] = 1.0
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        W = np.zeros((Z.shape[1], len(MOVES)))
        n = len(X)
        for _ in range(self.iters):
            P = _softmax(Z @ W)
            grad = Z.T @ (P - Y) / n
            grad[1:] += self.l2 * W[1:] / n
            W -= self.lr * grad
        self.W, self.n_train = W, n
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("model not fitted")
        X = np.atleast_2d(X)
        Z = np.hstack([np.ones((len(X), 1)), (X - self.mu) / self.sd])
        return _softmax(Z @ self.W)

    def predict_state(self, s: CompanyState) -> dict[str, float]:
        return dict(zip(MOVES, self.predict(features(s))[0]))


def _softmax(A: np.ndarray) -> np.ndarray:
    A = A - A.max(1, keepdims=True)
    E = np.exp(A)
    return E / E.sum(1, keepdims=True)


def brier(P: np.ndarray, y: list[str]) -> float:
    Y = np.zeros_like(P)
    for i, m in enumerate(y):
        Y[i, MOVES.index(m)] = 1.0
    return float(((P - Y) ** 2).sum(1).mean())


def fit_asof(panel: list[dict], as_of: str, **kw) -> MoveModel | None:
    """Fit only on histories whose outcome was KNOWN by as_of.

    Each panel row carries the state date and the date its outcome became observable
    (the move's filing, or the end of the horizon). Using a row before that second date
    would let the model learn from the future: the most common silent error in event studies.
    """
    rows = [r for r in panel if r["outcome_known"] <= as_of]
    if len(rows) < 40 or len({r["move"] for r in rows}) < 2:
        return None
    return MoveModel(**kw).fit(np.vstack([features(r["state"]) for r in rows]), [r["move"] for r in rows])


# ------------------------------------------------------------------------- payoffs
@dataclass
class PayoffBook:
    """What each move has paid, as a multiple of the entry price. Learned walk-forward.

    CASH_RETURN is valued structurally (cash reaching holders per share), not from history,
    because its value depends on this company's own balance sheet.
    """
    reverse_merger: float = 0.70       # starting priors; replaced by medians from history
    dilutive_raise: float = 0.75
    status_quo: float = 0.90
    delist: float = 0.15
    recovery: float = 0.80             # share of adjusted net cash that reaches holders
    resolution_quarters: float = 3.0   # cash burns until a wind-down completes
    samples: dict = field(default_factory=dict)

    def learn(self, history: list[dict], as_of: str, min_n: int = 15) -> "PayoffBook":
        """history rows: {move, multiple, resolved, recovery?}. Uses only rows resolved by as_of."""
        known = [h for h in history if h["resolved"] <= as_of]
        for move, attr in ((REVERSE_MERGER, "reverse_merger"), (DILUTIVE_RAISE, "dilutive_raise"),
                           (STATUS_QUO, "status_quo"), (DELIST, "delist")):
            xs = [h["multiple"] for h in known if h["move"] == move]
            self.samples[move] = len(xs)
            if len(xs) >= min_n:
                setattr(self, attr, float(np.median(xs)))
        rec = [h["recovery"] for h in known if h["move"] == CASH_RETURN and h.get("recovery") is not None]
        self.samples[CASH_RETURN] = len(rec)
        if len(rec) >= min_n:
            self.recovery = float(np.median(rec))
        return self

    def cash_return_multiple(self, s: CompanyState) -> float:
        cash_left = s.net_cash - self.resolution_quarters * s.quarterly_burn
        per_share = max(cash_left, 0.0) * self.recovery / s.shares
        return per_share / s.price

    def multiple(self, move: str, s: CompanyState) -> float:
        return {CASH_RETURN: self.cash_return_multiple(s), REVERSE_MERGER: self.reverse_merger,
                DILUTIVE_RAISE: self.dilutive_raise, STATUS_QUO: self.status_quo, DELIST: self.delist}[move]


# ------------------------------------------------------------------------ lookahead
def after_move(s: CompanyState, move: str, raise_fraction: float = 0.3, raise_discount: float = 0.25) -> CompanyState:
    """The position after a non-terminal move, for the second ply of the search."""
    if move == DILUTIVE_RAISE:
        cap = s.shelf_capacity if math.isfinite(s.shelf_capacity) else raise_fraction * s.market_value
        proceeds = min(raise_fraction * s.market_value, cap)
        new_price = s.price * (1 - raise_discount)
        new_shares = s.shares + proceeds / new_price
        return replace(s, price=new_price, shares=new_shares, net_cash=s.net_cash + proceeds - 2 * s.quarterly_burn,
                       shelf_capacity=max(cap - proceeds, 0.0), atm_open=False)
    if move == STATUS_QUO:
        return replace(s, net_cash=s.net_cash - 4 * s.quarterly_burn, price=s.price * 0.9)
    raise ValueError(f"{move} is terminal")


@dataclass(frozen=True)
class Evaluation:
    expected_multiple: float           # expected value per 1 unit of entry price, before costs
    p_loss_over_half: float
    move_probs: dict
    line: list                         # (probability, path, multiple) for every leaf, best first
    reasons: list


def evaluate(s: CompanyState, model: MoveModel, payoffs: PayoffBook, depth: int = 2) -> Evaluation:
    """Expectimax over company moves to `depth` plies. Leaves are valued by the payoff book."""
    leaves: list[tuple[float, tuple, float]] = []

    def walk(state: CompanyState, prob: float, path: tuple, mult_so_far: float, d: int) -> None:
        probs = model.predict_state(state)
        for move, p in probs.items():
            pp = prob * p
            if pp < 1e-6:
                continue
            if move in TERMINAL or d == 1:
                leaves.append((pp, path + (move,), mult_so_far * payoffs.multiple(move, state)))
            else:
                nxt = after_move(state, move)
                walk(nxt, pp, path + (move,), mult_so_far * nxt.price / state.price, d - 1)

    walk(s, 1.0, (), 1.0, depth)
    total = sum(p for p, _, _ in leaves)
    ev = sum(p * m for p, _, m in leaves) / total
    p_half = sum(p for p, _, m in leaves if m < 0.5) / total
    leaves.sort(key=lambda t: -t[0])
    return Evaluation(ev, p_half, model.predict_state(s), leaves[:8], [])


# ------------------------------------------------------------------------- decision
@dataclass(frozen=True)
class Policy:
    """Every decision threshold in one place. Values are starting choices, frozen before testing."""
    burn_quarters: float = 2.0
    min_discount: float = 0.0            # market value below adjusted net cash
    require_trigger: bool = True
    hurdle: float = 0.25                 # expected multiple must exceed 1 + costs + hurdle
    max_p_loss_over_half: float = 0.15
    min_price: float = 1.00
    min_daily_value: float = 100_000.0
    round_trip_cost: float = 0.03        # replaced per name by the cost model when quotes exist
    max_position_fraction: float = 0.10  # of book equity
    kelly_fraction: float = 0.25         # fraction of the Kelly bet actually taken


@dataclass(frozen=True)
class Decision:
    act: bool
    size_fraction: float
    evaluation: Evaluation | None
    reasons: tuple


def decide(s: CompanyState, model: MoveModel | None, payoffs: PayoffBook, policy: Policy,
           daily_value: float) -> Decision:
    """Gate, evaluate, size. Every 'no' carries its reason; silence is never a decision."""
    why = []
    if s.price < policy.min_price:
        why.append(f"price {s.price:.2f} below {policy.min_price:.2f}")
    if daily_value < policy.min_daily_value:
        why.append(f"daily traded value {daily_value:,.0f} below {policy.min_daily_value:,.0f}")
    if s.discount(policy.burn_quarters) <= policy.min_discount:
        why.append("not below burn-adjusted net cash")
    if policy.require_trigger and not s.trigger():
        why.append("no cash-return trigger on file")
    if s.supply_overhang():
        why.append("supply overhang: " + ", ".join(s.supply_overhang()))
    if model is None:
        why.append("move model not yet trainable: too little resolved history")
    if why:
        return Decision(False, 0.0, None, tuple(why))
    ev = evaluate(s, model, payoffs)
    edge = ev.expected_multiple - 1.0 - policy.round_trip_cost
    if edge < policy.hurdle:
        return Decision(False, 0.0, ev, (f"expected edge {edge:.1%} below hurdle {policy.hurdle:.0%}",))
    if ev.p_loss_over_half > policy.max_p_loss_over_half:
        return Decision(False, 0.0, ev, (f"P(loss > 50%) {ev.p_loss_over_half:.0%} above {policy.max_p_loss_over_half:.0%}",))
    # Kelly for a single bet with leaf outcomes: maximise E[log(1 + f (m - 1 - c))] by grid search.
    outcomes = np.array([m - 1.0 - policy.round_trip_cost for _, _, m in ev.line])
    weights = np.array([p for p, _, _ in ev.line])
    weights = weights / weights.sum()
    grid = np.linspace(0, 1, 101)
    growth = [np.sum(weights * np.log(np.maximum(1 + f * outcomes, 1e-9))) for f in grid]
    kelly = float(grid[int(np.argmax(growth))])
    size = min(policy.kelly_fraction * kelly, policy.max_position_fraction)
    return Decision(size > 0, size, ev, (f"expected edge {edge:.1%}; full Kelly {kelly:.0%}; taking {size:.0%}",))
