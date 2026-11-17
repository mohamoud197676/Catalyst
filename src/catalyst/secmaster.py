"""Point-in-time security master.

Answers three questions a backtest must never get wrong:
  1. Which company did ticker X mean on date D?  (tickers are reused)
  2. Which securities were tradable on date D?    (including ones that later died)
  3. What is a price from date A worth in the share units of date B?  (splits, reverse splits)
"""
from __future__ import annotations

import sqlite3
from datetime import date

from .core import iso, utcnow


class AmbiguousTicker(LookupError):
    pass


def _d(value: date | str) -> str:
    return value if isinstance(value, str) else value.isoformat()


def create_security(conn: sqlite3.Connection, *, cik: int | None = None, cusip: str | None = None,
                    isin: str | None = None, description: str | None = None,
                    security_type: str = "CS") -> int:
    cur = conn.execute(
        "INSERT INTO security (cik, cusip, isin, security_type, description, created_at_utc) VALUES (?,?,?,?,?,?)",
        (cik, cusip, isin, security_type, description, iso(utcnow())))
    return int(cur.lastrowid)


def assign_ticker(conn: sqlite3.Connection, security_id: int, ticker: str, valid_from: date | str,
                  valid_to: date | str | None = None, exchange: str | None = None,
                  source: str = "MANUAL") -> None:
    """Record that security_id traded as ticker over [valid_from, valid_to).

    Refuses an overlap with another security on the same ticker: that would make
    point-in-time lookup ambiguous, and it almost always means a bad source row.
    """
    ticker = ticker.upper()
    vf, vt = _d(valid_from), (_d(valid_to) if valid_to else None)
    if vt is not None and vt <= vf:
        raise ValueError("valid_to must be after valid_from")
    clash = conn.execute(
        "SELECT security_id, valid_from, valid_to FROM ticker_history WHERE ticker=? AND security_id<>? "
        "AND valid_from < COALESCE(?, '9999-12-31') AND COALESCE(valid_to, '9999-12-31') > ?",
        (ticker, security_id, vt, vf)).fetchone()
    if clash:
        raise AmbiguousTicker(
            f"{ticker} already belongs to security {clash['security_id']} over "
            f"[{clash['valid_from']}, {clash['valid_to']}); close that range first")
    conn.execute(
        "INSERT INTO ticker_history (security_id, ticker, exchange, valid_from, valid_to, source, first_seen_at_utc) "
        "VALUES (?,?,?,?,?,?,?) ON CONFLICT(security_id, ticker, valid_from) DO UPDATE SET valid_to=excluded.valid_to",
        (security_id, ticker, exchange, vf, vt, source, iso(utcnow())))


def change_ticker(conn: sqlite3.Connection, security_id: int, new_ticker: str, effective: date | str,
                  source: str = "MANUAL", source_ref: str | None = None) -> None:
    eff = _d(effective)
    old = conn.execute(
        "SELECT id, ticker, exchange FROM ticker_history WHERE security_id=? AND valid_to IS NULL "
        "ORDER BY valid_from DESC LIMIT 1", (security_id,)).fetchone()
    if old is None:
        raise LookupError(f"security {security_id} has no open ticker range")
    conn.execute("UPDATE ticker_history SET valid_to=? WHERE id=?", (eff, old["id"]))
    assign_ticker(conn, security_id, new_ticker, eff, exchange=old["exchange"], source=source)
    add_corporate_action(conn, security_id, "TICKER_CHANGE", eff, old_value=old["ticker"],
                         new_value=new_ticker.upper(), source=source, source_ref=source_ref)


def resolve_ticker(conn: sqlite3.Connection, ticker: str, as_of: date | str) -> int | None:
    """security_id that traded as ticker on as_of, or None. Never guesses."""
    rows = conn.execute(
        "SELECT DISTINCT security_id FROM ticker_history WHERE ticker=? AND valid_from<=? "
        "AND COALESCE(valid_to, '9999-12-31')>?", (ticker.upper(), _d(as_of), _d(as_of))).fetchall()
    if len(rows) > 1:
        raise AmbiguousTicker(f"{ticker} maps to {len(rows)} securities on {_d(as_of)}")
    return int(rows[0]["security_id"]) if rows else None


def ticker_on(conn: sqlite3.Connection, security_id: int, as_of: date | str) -> str | None:
    row = conn.execute(
        "SELECT ticker FROM ticker_history WHERE security_id=? AND valid_from<=? "
        "AND COALESCE(valid_to, '9999-12-31')>? ORDER BY valid_from DESC LIMIT 1",
        (security_id, _d(as_of), _d(as_of))).fetchone()
    return row["ticker"] if row else None


def record_listing_event(conn: sqlite3.Connection, security_id: int, event_type: str, event_date: date | str,
                         *, reason: str | None = None, last_price: float | None = None,
                         successor_security_id: int | None = None, source: str = "MANUAL",
                         source_ref: str | None = None, close_ticker: bool = True) -> None:
    ed = _d(event_date)
    conn.execute(
        "INSERT OR IGNORE INTO listing_event (security_id, event_type, event_date, reason, last_price, "
        "successor_security_id, source, source_ref, first_seen_at_utc) VALUES (?,?,?,?,?,?,?,?,?)",
        (security_id, event_type, ed, reason, last_price, successor_security_id, source, source_ref, iso(utcnow())))
    if close_ticker and event_type in ("DELISTED", "ACQUIRED", "MERGED"):
        conn.execute("UPDATE ticker_history SET valid_to=? WHERE security_id=? AND valid_to IS NULL "
                     "AND valid_from<?", (ed, security_id, ed))


def add_corporate_action(conn: sqlite3.Connection, security_id: int, action_type: str, ex_date: date | str,
                         *, split_from: float | None = None, split_to: float | None = None,
                         cash_amount: float | None = None, old_value: str | None = None,
                         new_value: str | None = None, announced_at_utc: str | None = None,
                         source: str = "MANUAL", source_ref: str | None = None) -> None:
    if action_type in ("SPLIT", "REVERSE_SPLIT"):
        if not split_from or not split_to:
            raise ValueError("splits need split_from and split_to")
        is_reverse = split_to < split_from
        if is_reverse != (action_type == "REVERSE_SPLIT"):
            raise ValueError(f"{action_type} contradicts ratio {split_from}->{split_to}")
    conn.execute(
        "INSERT OR IGNORE INTO corporate_action (security_id, action_type, ex_date, announced_at_utc, split_from, "
        "split_to, cash_amount, old_value, new_value, source, source_ref, first_seen_at_utc) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
        (security_id, action_type, _d(ex_date), announced_at_utc, split_from, split_to, cash_amount,
         old_value, new_value, source, source_ref, iso(utcnow())))


def price_factor(conn: sqlite3.Connection, security_id: int, from_date: date | str, to_date: date | str,
                 known_by_utc: str | None = None) -> float:
    """Multiplier turning a price quoted on from_date into the share units of to_date.

    Splits with from_date < ex_date <= to_date apply. A 1-for-10 reverse split (from=10, to=1)
    gives 10.0: a $0.50 price before becomes $5.00 in post-split units.
    known_by_utc restricts to actions this system had seen by that time, so a backtest cannot
    use a split it had not yet heard about.
    """
    fd, td = _d(from_date), _d(to_date)
    if fd > td:
        return 1.0 / price_factor(conn, security_id, td, fd, known_by_utc)
    sql = ("SELECT ex_date, split_from, split_to FROM corporate_action WHERE security_id=? "
           "AND action_type IN ('SPLIT','REVERSE_SPLIT') AND ex_date>? AND ex_date<=?")
    args: list = [security_id, fd, td]
    if known_by_utc:
        sql += " AND first_seen_at_utc<=?"
        args.append(known_by_utc)
    factor, seen = 1.0, set()
    for row in conn.execute(sql + " ORDER BY ex_date", args):
        key = (row["ex_date"], row["split_from"], row["split_to"])
        # the same split reported by two sources must not be applied twice
        if key in seen:
            continue
        seen.add(key)
        factor *= row["split_from"] / row["split_to"]
    return factor


def universe_asof(conn: sqlite3.Connection, as_of: date | str) -> list[sqlite3.Row]:
    """Every security with a live ticker on as_of and no terminal event on or before it.

    This is the survivorship-free universe: a company delisted in 2023 is present for 2022.
    """
    d = _d(as_of)
    return conn.execute(
        "SELECT s.security_id, t.ticker, s.cik FROM security s "
        "JOIN ticker_history t ON t.security_id=s.security_id AND t.valid_from<=? "
        "     AND COALESCE(t.valid_to,'9999-12-31')>? "
        "LEFT JOIN v_terminal_event e ON e.security_id=s.security_id "
        "WHERE e.terminal_date IS NULL OR e.terminal_date>? ORDER BY t.ticker", (d, d, d)).fetchall()


def terminal_event(conn: sqlite3.Connection, security_id: int) -> sqlite3.Row | None:
    return conn.execute(
        "SELECT * FROM listing_event WHERE security_id=? AND event_type IN "
        "('DELISTED','ACQUIRED','MERGED','DEREGISTERED') ORDER BY event_date LIMIT 1", (security_id,)).fetchone()
