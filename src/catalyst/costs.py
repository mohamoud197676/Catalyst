"""Trading cost model, direction-aware.

Design rules
  * A trade the model cannot price is REJECTED, never priced at zero. No quote, no volume,
    no borrow record: no trade.
  * Every input that has not been verified against a source is named in the output.
  * Where a fee schedule is ambiguous, the costlier reading is used.
This module knows nothing about the project's return target.
"""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import asdict, dataclass, field
from datetime import datetime

from .core import iso, parse_ts, utcnow

SELL_ACTIONS = {"SELL", "SELL_SHORT"}
ACTIONS = {"BUY", "SELL", "SELL_SHORT", "BUY_TO_COVER"}


@dataclass
class TradeInputs:
    ticker: str
    action: str                       # BUY | SELL | SELL_SHORT | BUY_TO_COVER
    qty: float
    price: float                      # expected fill price before costs
    on_date: str                      # YYYY-MM-DD, selects the fee schedule in force
    bid: float | None = None
    ask: float | None = None
    spread_bps_estimate: float | None = None   # used only when no quote exists; flagged unverified
    window_volume: float | None = None         # shares traded in the window the order would work in
    daily_volatility: float | None = None      # daily return standard deviation, as a fraction
    hold_days: float = 1.0                     # calendar days the position is expected to stay open
    borrow: dict | None = None                 # {fee_rate_pct, shortable_shares, captured_at_utc}
    as_of_utc: str | None = None               # decision time, for borrow staleness
    plan: str = "FIXED"                        # IBKR pricing plan
    convert_fx: bool = False                   # True when GBP is converted for this trade
    max_fee_rate_pct: float | None = None
    borrow_max_age_minutes: float | None = None


@dataclass
class CostEstimate:
    status: str
    ticker: str
    action: str
    qty: float
    notional_usd: float
    components: dict = field(default_factory=dict)
    total_usd: float | None = None
    total_bps: float | None = None
    participation: float | None = None
    unverified_inputs: list = field(default_factory=list)
    notes: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == "OK"


class FeeBook:
    """Fee schedule lookup by date, remembering which unverified rows were used."""

    def __init__(self, conn: sqlite3.Connection, on_date: str, plan: str) -> None:
        self.conn, self.on_date, self.plan = conn, on_date, plan
        self.unverified: list[str] = []

    def get(self, broker: str, component: str, default: float | None = None) -> float:
        row = self.conn.execute(
            "SELECT value, verified FROM fee_schedule WHERE broker=? AND component=? AND plan IN (?, 'ALL') "
            "AND effective_from<=? AND (effective_to IS NULL OR effective_to>?) ORDER BY effective_from DESC LIMIT 1",
            (broker, component, self.plan, self.on_date, self.on_date)).fetchone()
        if row is None:
            if default is None:
                raise LookupError(f"no fee row for {broker}/{self.plan}/{component} on {self.on_date}")
            self.unverified.append(f"{component} (no schedule row; default {default})")
            return default
        if not row["verified"]:
            self.unverified.append(component)
        return float(row["value"])


def _reject(est: CostEstimate, reason: str, note: str) -> CostEstimate:
    est.status = reason
    est.notes.append(note)
    return est


def estimate(conn: sqlite3.Connection, t: TradeInputs, log: bool = True) -> CostEstimate:
    if t.action not in ACTIONS:
        raise ValueError(f"unknown action {t.action!r}")
    if t.qty <= 0 or t.price <= 0:
        raise ValueError("qty and price must be positive")
    notional = t.qty * t.price
    est = CostEstimate("OK", t.ticker.upper(), t.action, t.qty, notional)
    fees = FeeBook(conn, t.on_date, t.plan)
    c = est.components

    # --- liquidity gates first: a trade that cannot be executed has no cost, it has no existence
    if t.window_volume is None or t.window_volume <= 0:
        return _finish(conn, _reject(est, "REJECT_NO_VOLUME", "no traded volume known for the execution window"), fees, log)
    est.participation = t.qty / t.window_volume
    max_part = fees.get("MODEL", "max_participation")
    if est.participation > max_part:
        return _finish(conn, _reject(est, "REJECT_PARTICIPATION",
                       f"order is {est.participation:.1%} of window volume; limit {max_part:.1%}"), fees, log)

    # --- borrow gate for new shorts
    if t.action == "SELL_SHORT":
        b = t.borrow
        if not b:
            return _finish(conn, _reject(est, "REJECT_NO_BORROW_RECORD",
                           "no borrow snapshot for this ticker; assumed unavailable"), fees, log)
        if t.as_of_utc and t.borrow_max_age_minutes is not None and b.get("captured_at_utc"):
            age_min = (parse_ts(t.as_of_utc) - parse_ts(b["captured_at_utc"])).total_seconds() / 60
            if age_min < 0:
                raise ValueError("borrow snapshot is from the future relative to as_of_utc: look-ahead")
            if age_min > t.borrow_max_age_minutes:
                return _finish(conn, _reject(est, "REJECT_STALE_BORROW",
                               f"borrow snapshot is {age_min:.0f} min old; limit {t.borrow_max_age_minutes:.0f}"), fees, log)
        if b.get("shortable_shares") is None or b["shortable_shares"] < t.qty:
            return _finish(conn, _reject(est, "REJECT_INSUFFICIENT_BORROW",
                           f"{b.get('shortable_shares')} shares shown available, {t.qty:.0f} needed"), fees, log)
        if b.get("fee_rate_pct") is None:
            return _finish(conn, _reject(est, "REJECT_NO_FEE", "borrow fee unknown"), fees, log)
        if t.max_fee_rate_pct is not None and b["fee_rate_pct"] > t.max_fee_rate_pct:
            return _finish(conn, _reject(est, "REJECT_FEE_ABOVE_CAP",
                           f"borrow fee {b['fee_rate_pct']:.1f}% a year exceeds cap {t.max_fee_rate_pct:.1f}%"), fees, log)

    # --- spread: crossing half the quoted spread
    if t.bid is not None and t.ask is not None:
        if t.ask < t.bid:
            raise ValueError("crossed quote: ask below bid")
        c["half_spread"] = (t.ask - t.bid) / 2 * t.qty
    elif t.spread_bps_estimate is not None:
        c["half_spread"] = notional * t.spread_bps_estimate / 2 / 1e4
        fees.unverified.append("spread (estimate, no quote)")
    else:
        return _finish(conn, _reject(est, "REJECT_NO_QUOTE", "no bid/ask and no spread estimate"), fees, log)

    # --- market impact: square-root law, cost = k * sigma * sqrt(participation) * notional
    if t.daily_volatility is None:
        return _finish(conn, _reject(est, "REJECT_NO_VOLATILITY", "daily volatility needed for impact"), fees, log)
    k = fees.get("MODEL", "impact_coefficient")
    c["impact"] = k * t.daily_volatility * math.sqrt(est.participation) * notional

    # --- commission
    per_share = fees.get("IBKR", "commission_per_share") * t.qty
    cap = fees.get("IBKR", "commission_max_pct_of_value") * notional
    minimum = fees.get("IBKR", "commission_min_per_order")
    c["commission"] = max(minimum, min(per_share, cap))
    if cap < minimum:
        est.notes.append("trade is small enough that the 1% cap is below the order minimum; minimum charged "
                         "(costlier reading of the schedule)")
    if t.plan == "TIERED":
        c["clearing"] = min(fees.get("IBKR", "clearing_per_share") * t.qty, 0.005 * notional)
        c["exchange"] = fees.get("IBKR", "exchange_per_share") * t.qty

    # --- regulatory pass-throughs
    c["cat_fee"] = fees.get("IBKR", "cat_fee_per_share") * t.qty
    if t.action in SELL_ACTIONS:
        c["sec_fee"] = fees.get("IBKR", "sec_fee_per_usd_sold") * notional
        c["finra_taf"] = fees.get("IBKR", "finra_taf_per_share_sold") * t.qty

    # --- borrow carry for the planned hold
    if t.action == "SELL_SHORT":
        markup = fees.get("MODEL", "borrow_collateral_markup")
        days = fees.get("MODEL", "borrow_day_count")
        c["borrow_fee"] = notional * markup * t.borrow["fee_rate_pct"] / 100 * max(t.hold_days, 1.0) / days
        est.notes.append("borrow fee is indicative and can be re-rated daily; availability can be recalled")

    if t.convert_fx:
        c["fx"] = notional * fees.get("MODEL", "fx_cost_bps") / 1e4

    return _finish(conn, est, fees, log)


def _finish(conn: sqlite3.Connection, est: CostEstimate, fees: FeeBook, log: bool) -> CostEstimate:
    est.unverified_inputs = sorted(set(fees.unverified))
    if est.ok:
        est.total_usd = sum(est.components.values())
        est.total_bps = est.total_usd / est.notional_usd * 1e4
    if log:
        conn.execute(
            "INSERT INTO cost_estimate_log (created_at_utc, ticker, action, qty, price, status, total_usd, total_bps, "
            "breakdown_json, unverified_inputs) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (iso(utcnow()), est.ticker, est.action, est.qty, est.notional_usd / est.qty, est.status, est.total_usd,
             est.total_bps, json.dumps({"components": est.components, "notes": est.notes}),
             json.dumps(est.unverified_inputs)))
    return est


def round_trip(conn: sqlite3.Connection, open_trade: TradeInputs, exit_price: float,
               exit_bid: float | None = None, exit_ask: float | None = None, log: bool = False) -> dict:
    """Cost of opening and closing. The exit reuses the entry's liquidity assumptions."""
    closing = {"BUY": "SELL", "SELL_SHORT": "BUY_TO_COVER"}[open_trade.action]
    entry = estimate(conn, open_trade, log=log)
    if not entry.ok:
        return {"status": entry.status, "entry": asdict(entry), "exit": None, "total_usd": None}
    close_trade = TradeInputs(**{**asdict(open_trade), "action": closing, "price": exit_price,
                                 "bid": exit_bid, "ask": exit_ask, "borrow": None, "hold_days": 0})
    if exit_bid is None and open_trade.bid is not None:
        spread = open_trade.ask - open_trade.bid
        close_trade.bid, close_trade.ask = exit_price - spread / 2, exit_price + spread / 2
    exit_est = estimate(conn, close_trade, log=log)
    total = entry.total_usd + (exit_est.total_usd or 0)
    return {"status": exit_est.status, "entry": asdict(entry), "exit": asdict(exit_est), "total_usd": total,
            "total_bps_of_entry": total / entry.notional_usd * 1e4,
            "unverified_inputs": sorted(set(entry.unverified_inputs) | set(exit_est.unverified_inputs))}
