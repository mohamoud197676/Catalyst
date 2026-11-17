"""Price bars with session tags, and measurement of price reaction to filings.

Bars are stored unadjusted. Every bar is tagged PRE, REG, POST or CLOSED from its own
timestamp, so pre-market and after-hours moves are first-class data, not noise to filter.
"""
from __future__ import annotations

import json
import os
import sqlite3
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from .core import ET, UTC, MarketClock, Session, SourceRegistry, iso, parse_ts, utcnow
from .secmaster import price_factor

METHOD_VERSION = "r1"
_SPAN_SECONDS = {"1m": 60, "5m": 300}


# ------------------------------------------------------------------------- storage
def store_bars(conn: sqlite3.Connection, security_id: int, bars: list[dict], timespan: str,
               source: str, clock: MarketClock) -> int:
    """Store bars shaped {t: datetime|epoch_ms, o, h, l, c, v, vw?, n?}. Idempotent per source."""
    now, new = iso(utcnow()), 0
    for b in bars:
        t = b["t"]
        start = t if isinstance(t, datetime) else datetime.fromtimestamp(t / 1000, UTC)
        if not (b["l"] <= min(b["o"], b["c"]) and b["h"] >= max(b["o"], b["c"])) or b["v"] < 0:
            raise ValueError(f"inconsistent bar refused: {b}")
        session = "DAY" if timespan == "1d" else clock.session_of(start).value
        cur = conn.execute(
            "INSERT OR IGNORE INTO price_bar VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (security_id, timespan, iso(start), session, b["o"], b["h"], b["l"], b["c"], b["v"],
             b.get("vw"), b.get("n"), source, now))
        new += cur.rowcount
    return new


class MassiveClient:
    """Aggregates from Massive's REST API. Endpoint confirmed in docs; never run live here.

    Bars cover pre-market, regular and after-hours sessions per Massive's documentation.
    Always requested unadjusted: adjustment is this system's job, from corporate_action.
    """

    BASE = "https://api.massive.com"

    def __init__(self, api_key: str | None = None, registry: SourceRegistry | None = None, opener=None) -> None:
        self.api_key = api_key or os.environ.get("MASSIVE_API_KEY")
        self.registry = registry or SourceRegistry()
        self._open = opener or urllib.request.urlopen

    def aggregates_url(self, ticker: str, multiplier: int, timespan: str, start: str, end: str) -> str:
        path = f"/v2/aggs/ticker/{urllib.parse.quote(ticker)}/range/{multiplier}/{timespan}/{start}/{end}"
        return f"{self.BASE}{path}?adjusted=false&sort=asc&limit=50000"

    def aggregates(self, ticker: str, multiplier: int, timespan: str, start: str, end: str) -> list[dict]:
        self.registry.require("massive_rest")
        if not self.api_key:
            raise RuntimeError("MASSIVE_API_KEY is not set")
        req = urllib.request.Request(self.aggregates_url(ticker, multiplier, timespan, start, end),
                                     headers={"Authorization": f"Bearer {self.api_key}"})
        with self._open(req, timeout=30) as resp:
            return json.loads(resp.read()).get("results") or []


# ----------------------------------------------------------------------- reactions
def _last_bar_ending_by(conn, security_id: int, ts: datetime, not_before: datetime | None = None):
    """Latest 1m bar that had fully completed by ts."""
    cutoff = iso(ts - timedelta(seconds=60))
    sql = "SELECT * FROM price_bar WHERE security_id=? AND timespan='1m' AND bar_start_utc<=?"
    args: list = [security_id, cutoff]
    if not_before is not None:
        sql += " AND bar_start_utc>=?"
        args.append(iso(not_before))
    return conn.execute(sql + " ORDER BY bar_start_utc DESC LIMIT 1", args).fetchone()


def _first_bar_from(conn, security_id: int, ts: datetime, within: timedelta, session: str | None = None):
    sql = "SELECT * FROM price_bar WHERE security_id=? AND timespan='1m' AND bar_start_utc>=? AND bar_start_utc<?"
    args: list = [security_id, iso(ts), iso(ts + within)]
    if session:
        sql += " AND session=?"
        args.append(session)
    return conn.execute(sql + " ORDER BY bar_start_utc LIMIT 1", args).fetchone()


def _daily_bar(conn, security_id: int, day: str):
    return conn.execute(
        "SELECT * FROM price_bar WHERE security_id=? AND timespan='1d' AND substr(bar_start_utc,1,10)=? LIMIT 1",
        (security_id, day)).fetchone()


def _window_end(window: str, anchor: datetime, effective_start: datetime, clock: MarketClock) -> datetime | None:
    if window.endswith("m") and window[:-1].isdigit():
        return effective_start + timedelta(minutes=int(window[:-1]))
    if window == "session_end":
        return clock.session_end(effective_start)
    if window == "next_open":
        return clock.next_regular_open(anchor)
    if window == "next_close":
        return clock.next_regular_close(anchor)
    if window.endswith("d_close"):
        return clock.nth_close_after(anchor, int(window.split("d")[0]))
    raise ValueError(f"unknown window {window!r}")


def _measure(conn, security_id: int, anchor: datetime, windows: list[str], clock: MarketClock,
             max_stale_s: int) -> dict[str, dict]:
    session = clock.session_of(anchor)
    # A filing accepted while the market is shut cannot be traded until the next extended open.
    if session is Session.CLOSED:
        nxt = clock.next_regular_open(anchor).astimezone(ET).date()
        effective_start = max(anchor, clock.extended_bounds(nxt)[0])
    else:
        effective_start = anchor
    ref = _last_bar_ending_by(conn, security_id, anchor)
    out: dict[str, dict] = {}
    for w in windows:
        res = {"anchor_session": session.value, "status": "OK", "ref_price": None, "ref_bar_utc": None,
               "ref_staleness_s": None, "end_utc": None, "end_price": None, "end_session": None, "ret": None,
               "volume_in_window": None, "n_bars": 0}
        out[w] = res
        if ref is None:
            res["status"] = "NO_REF_PRICE"
            continue
        ref_end = parse_ts(ref["bar_start_utc"]) + timedelta(seconds=60)
        res.update(ref_price=ref["close"], ref_bar_utc=ref["bar_start_utc"],
                   ref_staleness_s=int((anchor - ref_end).total_seconds()))
        end_ts = _window_end(w, anchor, effective_start, clock)
        if end_ts is None:
            res["status"] = "NO_END_PRICE"
            continue
        res["end_utc"] = iso(end_ts)
        ref_day = parse_ts(ref["bar_start_utc"]).astimezone(ET).date().isoformat()
        end_day = end_ts.astimezone(ET).date().isoformat()
        end_price = end_session = None
        if w == "next_open":
            bar = _first_bar_from(conn, security_id, end_ts, timedelta(minutes=30), "REG")
            daily = _daily_bar(conn, security_id, end_day)
            if bar:
                end_price, end_session = bar["open"], "REG"
            elif daily:
                end_price, end_session = daily["open"], "DAY"
        elif w in ("next_close",) or w.endswith("d_close"):
            daily = _daily_bar(conn, security_id, end_day)
            bar = _last_bar_ending_by(conn, security_id, end_ts, not_before=end_ts - timedelta(minutes=30))
            if daily:
                end_price, end_session = daily["close"], "DAY"
            elif bar:
                end_price, end_session = bar["close"], bar["session"]
        else:
            bar = _last_bar_ending_by(conn, security_id, end_ts, not_before=anchor)
            if bar:
                end_price, end_session = bar["close"], bar["session"]
        agg = conn.execute(
            "SELECT COUNT(*) AS n, COALESCE(SUM(volume),0) AS v FROM price_bar WHERE security_id=? AND timespan='1m' "
            "AND bar_start_utc>=? AND bar_start_utc<?", (security_id, iso(anchor), iso(end_ts))).fetchone()
        res.update(n_bars=agg["n"], volume_in_window=agg["v"])
        if end_price is None:
            res["status"] = "NO_END_PRICE"     # nothing traded in the window: itself a finding for illiquid names
            continue
        # Put the reference price into the share units of the end date before comparing,
        # so a reverse split between the two is not read as a 900% gain.
        factor = price_factor(conn, security_id, ref_day, end_day)
        res.update(end_price=end_price, end_session=end_session, ret=end_price / (ref["close"] * factor) - 1.0)
        if res["ref_staleness_s"] > max_stale_s:
            res["status"] = "STALE_REF"
    return out


def measure_reaction(conn: sqlite3.Connection, accession_no: str, security_id: int, clock: MarketClock,
                     settings: dict, benchmark_security_id: int | None = None) -> list[dict]:
    """Measure and store price reaction to one filing over every configured window and latency.

    latency 0 is the theoretical move from the last pre-filing price. Positive latencies
    measure what was still on the table for someone who needed that long to read and act.
    """
    row = conn.execute("SELECT accepted_at_utc FROM filing WHERE accession_no=?", (accession_no,)).fetchone()
    if row is None or row["accepted_at_utc"] is None:
        raise LookupError(f"filing {accession_no} has no acceptance timestamp")
    accepted = parse_ts(row["accepted_at_utc"])
    stored, now = [], iso(utcnow())
    for latency in settings["latencies_s"]:
        anchor = accepted + timedelta(seconds=latency)
        own = _measure(conn, security_id, anchor, settings["windows"], clock, settings["max_ref_staleness_s"])
        bench = (_measure(conn, benchmark_security_id, anchor, settings["windows"], clock,
                          settings["max_ref_staleness_s"]) if benchmark_security_id else {})
        for w, r in own.items():
            b_ret = bench.get(w, {}).get("ret")
            abnormal = r["ret"] - b_ret if (r["ret"] is not None and b_ret is not None) else None
            conn.execute(
                "INSERT OR REPLACE INTO filing_reaction VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (accession_no, security_id, w, latency, METHOD_VERSION, iso(anchor), r["anchor_session"],
                 r["status"], r["ref_price"], r["ref_bar_utc"], r["ref_staleness_s"], r["end_utc"],
                 r["end_price"], r["end_session"], r["ret"], b_ret, abnormal, r["volume_in_window"],
                 r["n_bars"], now))
            stored.append({"window": w, "latency_s": latency, **r, "benchmark_ret": b_ret, "abnormal_ret": abnormal})
    return stored
