"""Point-in-time fundamentals from the SEC XBRL company-facts feed, and the H1 signal inputs.

Endpoint (documented by the SEC): https://data.sec.gov/api/xbrl/companyfacts/CIK##########.json
The response layout used here (facts > taxonomy > tag > units > unit > [{end, val, accn, form,
filed, start?}]) is from working knowledge, NOT yet confirmed against a live response: the
parser skips anything that does not fit and reports how much it skipped.

Tag choices below are approximations. Companies tag cash, investments and debt inconsistently,
so every figure used in a test must be spot-checked against the filing for a hand sample.
"""
from __future__ import annotations

import sqlite3
from datetime import date

CASH_TAGS = ("CashAndCashEquivalentsAtCarryingValue",
             "CashCashEquivalentsRestrictedCashAndRestrictedCashEquivalents")
INVESTMENT_TAGS = ("ShortTermInvestments", "MarketableSecuritiesCurrent",
                   "AvailableForSaleSecuritiesDebtSecuritiesCurrent")
DEBT_TOTAL_TAGS = ("LongTermDebt", "DebtLongtermAndShorttermCombinedAmount")
DEBT_PART_TAGS = ("LongTermDebtNoncurrent", "LongTermDebtCurrent", "ShortTermBorrowings",
                  "ConvertibleNotesPayable", "NotesPayable")
OPERATING_CASH_FLOW = "NetCashProvidedByUsedInOperatingActivities"
SHARES_TAG = ("dei", "EntityCommonStockSharesOutstanding")
MAX_STALENESS_DAYS = 200      # a balance sheet older than this is treated as missing


def _d(v: date | str) -> str:
    return v if isinstance(v, str) else v.isoformat()


def store_companyfacts(conn: sqlite3.Connection, payload: dict) -> dict:
    """Store every well-formed fact. Returns counts so silent data loss is visible."""
    cik = int(payload["cik"])
    stored = skipped = 0
    for taxonomy, tags in (payload.get("facts") or {}).items():
        for tag, body in tags.items():
            for unit, rows in (body.get("units") or {}).items():
                for r in rows:
                    try:
                        args = (cik, taxonomy, tag, unit, r.get("start") or "", r["end"], float(r["val"]),
                                r["filed"], r.get("form"), r["accn"])
                    except (KeyError, TypeError, ValueError):
                        skipped += 1
                        continue
                    stored += conn.execute("INSERT OR IGNORE INTO fundamental_fact VALUES (?,?,?,?,?,?,?,?,?,?)",
                                           args).rowcount
    return {"cik": cik, "stored": stored, "skipped": skipped}


def _latest_instant(conn, cik: int, tags: tuple[str, ...], as_of: str, taxonomy: str = "us-gaap",
                    unit: str = "USD") -> tuple[float, str] | None:
    """Most recent period-end value among tags, using only filings made on or before as_of.

    Where a period was restated, the latest filing known by as_of wins.
    """
    best = None
    for tag in tags:
        row = conn.execute(
            "SELECT value, period_end FROM fundamental_fact WHERE cik=? AND taxonomy=? AND tag=? AND unit=? "
            "AND period_start='' AND filed_date<=? ORDER BY period_end DESC, filed_date DESC LIMIT 1",
            (cik, taxonomy, tag, unit, as_of)).fetchone()
        if row and (best is None or row["period_end"] > best[1]):
            best = (row["value"], row["period_end"])
    return best


def _sum_at(conn, cik: int, tags: tuple[str, ...], period_end: str, as_of: str) -> float:
    total = 0.0
    for tag in tags:
        row = conn.execute(
            "SELECT value FROM fundamental_fact WHERE cik=? AND taxonomy='us-gaap' AND tag=? AND unit='USD' "
            "AND period_start='' AND period_end=? AND filed_date<=? ORDER BY filed_date DESC LIMIT 1",
            (cik, tag, period_end, as_of)).fetchone()
        total += row["value"] if row else 0.0
    return total


def net_cash_asof(conn: sqlite3.Connection, cik: int, as_of: date | str) -> dict | None:
    """Cash plus short-term investments minus debt, from the latest balance sheet public on as_of.

    Debt is the larger of the reported total and the sum of its parts: when tagging is
    ambiguous the reading that makes the company look worse is used.
    """
    a = _d(as_of)
    cash = _latest_instant(conn, cik, CASH_TAGS, a)
    if cash is None:
        return None
    bs_date = cash[1]
    if (date.fromisoformat(a) - date.fromisoformat(bs_date)).days > MAX_STALENESS_DAYS:
        return None
    investments = max((_sum_at(conn, cik, (t,), bs_date, a) for t in INVESTMENT_TAGS), default=0.0)
    debt = max(max((_sum_at(conn, cik, (t,), bs_date, a) for t in DEBT_TOTAL_TAGS), default=0.0),
               _sum_at(conn, cik, DEBT_PART_TAGS, bs_date, a))
    return {"balance_sheet_date": bs_date, "cash": cash[0], "investments": investments, "debt": debt,
            "net_cash": cash[0] + investments - debt}


def quarterly_operating_cash_flows(conn: sqlite3.Connection, cik: int, as_of: date | str) -> list[tuple[str, float]]:
    """Single-quarter operating cash flows, newest first, built from year-to-date figures.

    Cash-flow statements report cumulative year-to-date amounts, so a quarter is the difference
    between two cumulative figures that share a start date.
    """
    rows = conn.execute(
        "SELECT period_start, period_end, value, MAX(filed_date) FROM fundamental_fact WHERE cik=? "
        "AND taxonomy='us-gaap' AND tag=? AND unit='USD' AND period_start<>'' AND filed_date<=? "
        "GROUP BY period_start, period_end", (cik, OPERATING_CASH_FLOW, _d(as_of))).fetchall()
    by_start: dict[str, list[tuple[str, float]]] = {}
    for r in rows:
        by_start.setdefault(r["period_start"], []).append((r["period_end"], r["value"]))
    quarters: dict[str, float] = {}
    for start, series in by_start.items():
        series.sort()
        prev_end, prev_val = start, 0.0
        for end, val in series:
            days = (date.fromisoformat(end) - date.fromisoformat(prev_end)).days
            if 75 <= days <= 105:                       # only accept a clean single-quarter step
                quarters.setdefault(end, val - prev_val)
            prev_end, prev_val = end, val
    return sorted(quarters.items(), reverse=True)


def quarterly_burn_asof(conn: sqlite3.Connection, cik: int, as_of: date | str) -> float | None:
    """Average cash used in operations over the two most recent quarters; 0 if cash-positive."""
    q = quarterly_operating_cash_flows(conn, cik, as_of)
    if len(q) < 2:
        return None
    return max(0.0, -(q[0][1] + q[1][1]) / 2)


def shares_asof(conn: sqlite3.Connection, cik: int, as_of: date | str) -> float | None:
    row = conn.execute(
        "SELECT value FROM fundamental_fact WHERE cik=? AND taxonomy=? AND tag=? AND unit='shares' AND filed_date<=? "
        "ORDER BY period_end DESC, filed_date DESC LIMIT 1", (cik, *SHARES_TAG, _d(as_of))).fetchone()
    return row["value"] if row else None


def h1_signal(conn: sqlite3.Connection, cik: int, as_of: date | str, price: float,
              split_factor_since_cover: float = 1.0, burn_quarters: float = 2.0) -> dict:
    """The H1 signal exactly as pre-registered: market value below net cash minus two quarters of burn.

    Any missing input returns status MISSING_INPUT and signal False. Unknown is never a pass.
    split_factor_since_cover converts the cover-page share count into today's share units.
    """
    nc, burn, shares = net_cash_asof(conn, cik, as_of), quarterly_burn_asof(conn, cik, as_of), shares_asof(conn, cik, as_of)
    if nc is None or burn is None or shares is None or price <= 0:
        return {"status": "MISSING_INPUT", "signal": False,
                "missing": [n for n, v in (("net_cash", nc), ("burn", burn), ("shares", shares)) if v is None]}
    market_value = price * shares / split_factor_since_cover
    adjusted = nc["net_cash"] - burn_quarters * burn
    return {"status": "OK", "signal": market_value < adjusted, "market_value": market_value,
            "net_cash": nc["net_cash"], "quarterly_burn": burn, "adjusted_net_cash": adjusted,
            "balance_sheet_date": nc["balance_sheet_date"],
            "discount": 1 - market_value / adjusted if adjusted > 0 else None}
