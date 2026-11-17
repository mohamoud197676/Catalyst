"""Borrow availability logger and free historical proxies.

Why this exists: historical stock-borrow data is the one input a retail researcher cannot
download. A short backtest without it assumes every stock could be borrowed, which is false
exactly for the stocks that fall the most. So the system builds its own history, starting
now, and uses public series as labelled proxies for the years before.

Nothing here places, modifies or requests a locate. It only reads public files.
"""
from __future__ import annotations

import csv
import ftplib
import io
import re
import sqlite3
from dataclasses import dataclass
from datetime import date

from ..core import IngestRun, SourceRegistry, iso, utcnow
from ..secmaster import AmbiguousTicker, resolve_ticker

# Column names the parser will accept, mapped to our fields. The file is read by header,
# not by position, because the published layout is documented only as an image.
_HEADER_ALIASES = {
    "SYM": "ticker", "SYMBOL": "ticker",
    "CUR": "currency", "CURRENCY": "currency",
    "NAME": "name",
    "CON": "conid", "CONID": "conid",
    "ISIN": "isin",
    "REBATERATE": "rebate", "REBATE RATE": "rebate",
    "FEERATE": "fee", "FEE RATE": "fee",
    "AVAILABLE": "available",
}
_REQUIRED = {"ticker", "available"}


class UnknownBorrowLayout(ValueError):
    pass


@dataclass(frozen=True)
class BorrowRow:
    ticker: str
    currency: str | None
    name: str | None
    conid: str | None
    isin: str | None
    rebate_rate_pct: float | None
    fee_rate_pct: float | None
    shortable_shares: int | None
    shortable_is_floor: bool
    raw_available: str


def _num(text: str | None) -> float | None:
    if text is None:
        return None
    t = text.strip()
    if t in ("", "NA", "N/A", "-"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def parse_available(raw: str) -> tuple[int | None, bool]:
    """'>10000000' -> (10000000, True): at least that many. '2500' -> (2500, False)."""
    t = raw.strip().replace(",", "")
    floor = t.startswith(">")
    t = t.lstrip(">").strip()
    if not re.fullmatch(r"\d+", t):
        return None, False
    return int(t), floor


def parse_shortstock_file(text: str) -> tuple[dict, list[BorrowRow]]:
    """Parse an IBKR short-stock availability file.

    Expected shape (pipe-delimited): comment lines beginning '#', one of which is the column
    header ('#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|'), an optional '#BOF|date|time'
    stamp, and an optional '#EOF|rowcount' trailer. If no header naming at least the ticker
    and available columns is found, the file is refused: a guessed layout would silently
    log wrong numbers for months.
    """
    meta: dict = {"source_asof_raw": None, "declared_rows": None}
    columns: list[str | None] | None = None
    rows: list[BorrowRow] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("#"):
            parts = [p.strip() for p in line.lstrip("#").split("|")]
            tag = parts[0].upper()
            if tag == "BOF":
                meta["source_asof_raw"] = " ".join(p for p in parts[1:3] if p)
            elif tag == "EOF":
                meta["declared_rows"] = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else None
            else:
                mapped = [_HEADER_ALIASES.get(p.upper()) for p in parts]
                if _REQUIRED <= {m for m in mapped if m}:
                    columns = mapped
            continue
        if columns is None:
            raise UnknownBorrowLayout("data row found before a recognised header line")
        parts = [p.strip() for p in line.split("|")]
        rec = {name: parts[i] for i, name in enumerate(columns) if name and i < len(parts)}
        if not rec.get("ticker"):
            continue
        shares, floor = parse_available(rec.get("available", ""))
        rows.append(BorrowRow(
            ticker=rec["ticker"].upper(), currency=rec.get("currency") or None, name=rec.get("name") or None,
            conid=rec.get("conid") or None, isin=rec.get("isin") or None,
            rebate_rate_pct=_num(rec.get("rebate")), fee_rate_pct=_num(rec.get("fee")),
            shortable_shares=shares, shortable_is_floor=floor, raw_available=rec.get("available", "")))
    if columns is None:
        raise UnknownBorrowLayout("no header line with ticker and available columns")
    if meta["declared_rows"] is not None and meta["declared_rows"] != len(rows):
        meta["row_count_mismatch"] = True      # truncated download: caller records the run as suspect
    return meta, rows


def fetch_ibkr_file(region: str = "usa", registry: SourceRegistry | None = None,
                    host: str = "ftp2.interactivebrokers.com", timeout: int = 60) -> str:
    """Download <region>.txt from IBKR's public FTP (user 'shortstock', empty password)."""
    (registry or SourceRegistry()).require("ibkr_shortstock_ftp", "PARTIAL")
    buf = io.BytesIO()
    with ftplib.FTP(host, timeout=timeout) as ftp:
        ftp.login("shortstock", "")
        ftp.retrbinary(f"RETR {region}.txt", buf.write)
    return buf.getvalue().decode("latin-1")


def log_snapshot(conn: sqlite3.Connection, text: str, region: str = "usa",
                 source_key: str = "ibkr_shortstock_ftp", captured_at=None) -> dict:
    """Append one full snapshot. Rows are never updated: the history is the product."""
    captured = captured_at or utcnow()
    with IngestRun(conn, source_key, {"region": region}) as run:
        meta, rows = parse_shortstock_file(text)
        if meta.get("row_count_mismatch"):
            raise ValueError(f"file declares {meta['declared_rows']} rows but {len(rows)} parsed: truncated download")
        unresolved = 0
        payload = []
        for r in rows:
            try:
                sid = resolve_ticker(conn, r.ticker, captured.date())
            except AmbiguousTicker:
                sid = None
            unresolved += sid is None
            payload.append((run.run_id, source_key, region, meta["source_asof_raw"], iso(captured), r.ticker, sid,
                            r.currency, r.name, r.conid, r.isin, r.fee_rate_pct, r.rebate_rate_pct,
                            r.shortable_shares, int(r.shortable_is_floor), r.raw_available))
        conn.executemany(
            "INSERT INTO borrow_snapshot (run_id, source_key, region, source_asof_raw, captured_at_utc, ticker, "
            "security_id, currency, name, conid, isin, fee_rate_pct, rebate_rate_pct, shortable_shares, "
            "shortable_is_floor, raw_available) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", payload)
        run.rows_seen = run.rows_new = len(rows)
        return {"rows": len(rows), "unresolved_tickers": unresolved, "source_asof_raw": meta["source_asof_raw"],
                "captured_at_utc": iso(captured)}


def borrow_asof(conn: sqlite3.Connection, ticker: str, as_of_utc: str) -> sqlite3.Row | None:
    """Latest snapshot for ticker captured at or before as_of_utc. Never looks ahead."""
    return conn.execute(
        "SELECT * FROM borrow_snapshot WHERE ticker=? AND captured_at_utc<=? ORDER BY captured_at_utc DESC LIMIT 1",
        (ticker.upper(), as_of_utc)).fetchone()


def borrow_history(conn: sqlite3.Connection, ticker: str) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT captured_at_utc, fee_rate_pct, rebate_rate_pct, shortable_shares, shortable_is_floor "
        "FROM borrow_snapshot WHERE ticker=? ORDER BY captured_at_utc", (ticker.upper(),)).fetchall()


# ------------------------------------------------------------------ historical proxies
def parse_ftd_file(text: str) -> list[dict]:
    """SEC fails-to-deliver file: SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE.

    Field list is from the SEC page. Price is '.' when unavailable. Trailer lines are skipped.
    """
    out = []
    for parts in csv.reader(io.StringIO(text), delimiter="|", quoting=csv.QUOTE_NONE):
        if len(parts) < 6 or not re.fullmatch(r"\d{8}", parts[0].strip()):
            continue
        d = parts[0].strip()
        qty = parts[3].strip()
        if not qty.isdigit():
            continue
        out.append({"as_of_date": f"{d[:4]}-{d[4:6]}-{d[6:]}", "cusip": parts[1].strip(),
                    "ticker": parts[2].strip().upper(), "quantity": float(qty),
                    "description": parts[4].strip(), "price": _num(parts[5])})
    return out


def store_ftd(conn: sqlite3.Connection, rows: list[dict], source_key: str = "sec_ftd") -> int:
    now, new = iso(utcnow()), 0
    with IngestRun(conn, source_key) as run:
        for r in rows:
            if not r["ticker"]:
                continue
            try:
                sid = resolve_ticker(conn, r["ticker"], r["as_of_date"])
            except AmbiguousTicker:
                sid = None
            cur = conn.execute(
                "INSERT OR IGNORE INTO borrow_proxy (proxy_type, as_of_date, ticker, cusip, security_id, quantity, "
                "price, description, source_key, ingested_at_utc) VALUES ('FTD',?,?,?,?,?,?,?,?,?)",
                (r["as_of_date"], r["ticker"], r["cusip"], sid, r["quantity"], r["price"], r["description"],
                 source_key, now))
            new += cur.rowcount
        run.rows_seen, run.rows_new = len(rows), new
    return new


def cusip_symbol_map(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    """Every CUSIP-symbol pairing seen in FTD data with its first and last date.

    A free identifier history that includes delisted names and exposes ticker reuse:
    one symbol appearing under two CUSIPs is two different securities.
    """
    return conn.execute(
        "SELECT cusip, ticker, MIN(as_of_date) AS first_seen, MAX(as_of_date) AS last_seen, "
        "MAX(description) AS description FROM borrow_proxy WHERE proxy_type='FTD' AND cusip<>'' "
        "GROUP BY cusip, ticker ORDER BY ticker, first_seen").fetchall()
