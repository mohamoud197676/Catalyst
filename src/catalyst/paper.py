"""Paper execution only.

There is no live broker adapter in this codebase, by design. Three independent locks:
  1. config: execution.mode must equal PAPER or the project refuses to load
  2. database: paper_order.mode carries CHECK (mode = 'PAPER')
  3. tests: tests/test_guards.py fails if any broker SDK or order-placement call appears in src/

Long and short run as separate books. A strategy belongs to exactly one book, and the
database refuses a short action in the long book and the reverse.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict

from .core import ProjectConfig, iso, utcnow
from .costs import TradeInputs, estimate

BOOK_ACTIONS = {"LONG": {"BUY", "SELL"}, "SHORT": {"SELL_SHORT", "BUY_TO_COVER"}}
EXECUTION_MODE = "PAPER"


def register_strategy(conn: sqlite3.Connection, strategy_id: str, book_id: str, name: str,
                      prereg_ref: str | None = None, config: dict | None = None) -> None:
    if book_id not in BOOK_ACTIONS:
        raise ValueError("book_id must be LONG or SHORT")
    conn.execute(
        "INSERT INTO strategy (strategy_id, book_id, name, prereg_ref, config_json, created_at_utc) "
        "VALUES (?,?,?,?,?,?)", (strategy_id, book_id, name, prereg_ref, json.dumps(config or {}), iso(utcnow())))


class PaperBroker:
    def __init__(self, conn: sqlite3.Connection, cfg: ProjectConfig) -> None:
        if cfg.execution_mode != EXECUTION_MODE:
            raise RuntimeError("PaperBroker only runs in PAPER mode")
        self.conn, self.cfg = conn, cfg

    def submit(self, *, client_key: str, strategy_id: str, action: str, ticker: str, qty: float,
               limit_price: float, trade: TradeInputs, security_id: int | None = None) -> dict:
        """Simulate an order. Returns the stored order; a repeated client_key returns the original."""
        existing = self.conn.execute("SELECT * FROM paper_order WHERE client_key=?", (client_key,)).fetchone()
        if existing:
            return {**dict(existing), "duplicate": True}
        strat = self.conn.execute("SELECT book_id FROM strategy WHERE strategy_id=?", (strategy_id,)).fetchone()
        if strat is None:
            raise LookupError(f"unknown strategy {strategy_id!r}")
        book_id = strat["book_id"]
        if action not in BOOK_ACTIONS[book_id]:
            raise PermissionError(f"strategy {strategy_id} is in the {book_id} book and cannot {action}")
        limits = self.cfg.book(book_id)
        trade.action, trade.qty, trade.ticker = action, qty, ticker
        if book_id == "SHORT":
            trade.max_fee_rate_pct = limits.get("max_fee_rate_pct")
            trade.borrow_max_age_minutes = limits.get("borrow_max_age_minutes")

        # marketable-limit fill rule: buys fill at the ask, sells at the bid, if the limit allows
        status, reason, fill_price = "ACCEPTED", None, None
        if trade.bid is None or trade.ask is None:
            status, reason = "REJECTED", "REJECT_NO_QUOTE"
        else:
            buying = action in ("BUY", "BUY_TO_COVER")
            touch = trade.ask if buying else trade.bid
            marketable = limit_price >= touch if buying else limit_price <= touch
            if marketable:
                trade.price = touch
                est = estimate(self.conn, trade)
                if est.ok:
                    status, fill_price = "FILLED", touch
                else:
                    status, reason = "REJECTED", est.status
        now = iso(utcnow())
        cur = self.conn.execute(
            "INSERT INTO paper_order (client_key, mode, strategy_id, book_id, security_id, ticker, action, qty, "
            "limit_price, created_at_utc, status, reject_reason) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (client_key, EXECUTION_MODE, strategy_id, book_id, security_id, ticker.upper(), action, qty,
             limit_price, now, status, reason))
        order_id = cur.lastrowid
        if status == "FILLED":
            # spread is already inside the fill price (we crossed it); do not charge it twice
            charged = {k: v for k, v in est.components.items() if k != "half_spread"}
            self.conn.execute(
                "INSERT INTO paper_fill (order_id, filled_at_utc, qty, price, cost_usd, cost_json) VALUES (?,?,?,?,?,?)",
                (order_id, now, qty, fill_price, sum(charged.values()),
                 json.dumps({"charged": charged, "spread_in_price": est.components.get("half_spread"),
                             "unverified_inputs": est.unverified_inputs})))
        return {**dict(self.conn.execute("SELECT * FROM paper_order WHERE order_id=?", (order_id,)).fetchone()),
                "duplicate": False}

    def book_summary(self, book_id: str) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) AS orders, SUM(status='FILLED') AS filled, SUM(status='REJECTED') AS rejected "
            "FROM paper_order WHERE book_id=?", (book_id,)).fetchone()
        costs = self.conn.execute(
            "SELECT COALESCE(SUM(f.cost_usd),0) AS c FROM paper_fill f JOIN paper_order o USING(order_id) "
            "WHERE o.book_id=?", (book_id,)).fetchone()
        return {"book": book_id, "orders": row["orders"], "filled": row["filled"] or 0,
                "rejected": row["rejected"] or 0, "explicit_costs_usd": costs["c"]}
