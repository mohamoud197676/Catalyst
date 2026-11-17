"""Step 3 of the run order: the free feasibility count for H1. No price data is used.

Counts trigger filings 2010-2025 at companies that, on the filing date, reported positive net
cash and negative operating cash flow (burning cash). This is an UPPER BOUND on H1 events:
the below-cash price test, the liquidity floor and the supply filter can only remove events.
If the bound is under 150, H1 is retired as untestable before any money is spent.

Triggers counted here: initial Schedule 13D and third-party tender offers (SC TO-T). Strategic
reviews and liquidation plans live inside 8-K and proxy text and are not counted, so the bound
is incomplete in that direction; results are reported per trigger type.

Subject company of a 13D: the master index lists a filing under every party. The subject is
taken to be the party that also files 10-K or 10-Q reports; a fund filer rarely does. This
heuristic is logged and checked on a sample.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import pathlib
import sqlite3
import sys
import time
from datetime import date

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from catalyst import core, fundamentals  # noqa: E402
from catalyst.edgar.client import EdgarClient  # noqa: E402
from catalyst.edgar.ingest import parse_master_index  # noqa: E402

TRIGGERS = {"SC 13D": "ACTIVIST_13D", "SCHEDULE 13D": "ACTIVIST_13D", "SC TO-T": "TENDER_OFFER",
            "SCHEDULE TO-T": "TENDER_OFFER"}
REPORT_FORMS = {"10-K", "10-Q", "10-K/A", "10-Q/A"}
OUT = pathlib.Path("results")


def quarter_index(client: EdgarClient, year: int, q: int) -> list[dict]:
    url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.idx"
    return parse_master_index(client.get(url).decode("latin-1"))


def main(start: int = 2010, end: int = 2025) -> int:
    ua = os.environ.get("SEC_USER_AGENT", "")
    client = EdgarClient(ua)
    OUT.mkdir(exist_ok=True)
    log: dict = {"started": core.iso(core.utcnow()), "years": [start, end], "index_rows": 0, "notes": []}

    reporters: set[int] = set()
    by_accession: dict[str, list[dict]] = collections.defaultdict(list)
    for year in range(start, end + 1):
        for q in range(1, 5):
            rows = quarter_index(client, year, q)
            log["index_rows"] += len(rows)
            for r in rows:
                form = r["form_type"].upper()
                if form in REPORT_FORMS:
                    reporters.add(r["cik"])
                if form in TRIGGERS and r["accession_no"]:
                    by_accession[r["accession_no"]].append(r)
        print(f"indexed {year}: {len(by_accession)} trigger filings so far", flush=True)

    candidates, ambiguous = [], 0
    for acc, parties in by_accession.items():
        subjects = {p["cik"] for p in parties if p["cik"] in reporters}
        if len(subjects) != 1:
            ambiguous += 1
            continue
        p = parties[0]
        candidates.append({"accession": acc, "cik": subjects.pop(), "trigger": TRIGGERS[p["form_type"].upper()],
                           "filed": p["filed_date"], "company": next(x["company"] for x in parties if x["cik"] in reporters)})
    log.update(trigger_filings=len(by_accession), subject_ambiguous=ambiguous, candidates=len(candidates))

    # one company-facts download per subject company
    facts_cache: dict[int, sqlite3.Connection | None] = {}
    events, missing = [], 0
    for i, c in enumerate(sorted(candidates, key=lambda x: (x["cik"], x["filed"]))):
        cik = c["cik"]
        if cik not in facts_cache:
            try:
                payload = json.loads(client.get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"))
                conn = core.connect(":memory:")
                conn.executescript(core.SCHEMA_PATH.read_text())
                fundamentals.store_companyfacts(conn, payload)
                facts_cache[cik] = conn
            except Exception as exc:  # noqa: BLE001 - absence of XBRL is common for old or foreign filers
                facts_cache[cik] = None
                log["notes"].append(f"no facts for {cik}: {exc!r}"[:200])
            if len(facts_cache) % 200 == 0:
                print(f"facts fetched for {len(facts_cache)} companies", flush=True)
        conn = facts_cache[cik]
        if conn is None:
            missing += 1
            continue
        nc = fundamentals.net_cash_asof(conn, cik, c["filed"])
        burn = fundamentals.quarterly_burn_asof(conn, cik, c["filed"])
        if nc is None or burn is None:
            missing += 1
            continue
        if nc["net_cash"] > 0 and burn > 0:
            adj = nc["net_cash"] - 2 * burn
            pf = conn.execute(
                "SELECT value, period_end FROM fundamental_fact WHERE cik=? AND taxonomy='dei' AND tag='EntityPublicFloat' "
                "AND filed_date<=? ORDER BY period_end DESC LIMIT 1", (cik, c["filed"])).fetchone()
            events.append({**c, "net_cash": nc["net_cash"], "quarterly_burn": burn,
                           "runway_q": nc["net_cash"] / burn, "bs_date": nc["balance_sheet_date"],
                           "adjusted_net_cash": adj,
                           "public_float": pf["value"] if pf else None, "float_date": pf["period_end"] if pf else None,
                           "below_cash_proxy": bool(pf and adj > 0 and pf["value"] < adj)})
        if len(facts_cache) > 400 and i % 50 == 0:   # keep memory bounded on a small runner
            for k in list(facts_cache)[:-50]:
                if facts_cache[k] is not None:
                    facts_cache[k].close()
                facts_cache[k] = facts_cache[k] and None
    # one event per company per 12 months, as the frozen rules require
    kept, last = [], {}
    for e in sorted(events, key=lambda x: (x["cik"], x["filed"])):
        prev = last.get(e["cik"])
        if prev is None or (date.fromisoformat(e["filed"]) - date.fromisoformat(prev)).days >= 365:
            kept.append(e)
            last[e["cik"]] = e["filed"]
    by_year = collections.Counter(e["filed"][:4] for e in kept)
    by_trig = collections.Counter(e["trigger"] for e in kept)
    holdout = sum(1 for e in kept if "2020" <= e["filed"][:4] <= "2025")
    proxy = [e for e in kept if e["below_cash_proxy"]]
    has_float = [e for e in kept if e["public_float"] is not None]
    log.update(positive_adjusted_cash=sum(e["adjusted_net_cash"] > 0 for e in kept),
               with_public_float=len(has_float), below_cash_proxy_total=len(proxy),
               below_cash_proxy_holdout=sum("2020" <= e["filed"][:4] <= "2025" for e in proxy),
               below_cash_proxy_by_year=dict(sorted(collections.Counter(e["filed"][:4] for e in proxy).items())),
               proxy_note=("Public float (non-affiliate market value, from the latest annual report cover known on the "
                           "filing date) understates market value, so this proxy over-counts below-cash events. "
                           "It is still an upper bound, tighter than the first."))
    log.update(fundamentals_missing=missing, burning_with_net_cash=len(events), upper_bound_events=len(kept),
               upper_bound_holdout=holdout, by_year=dict(sorted(by_year.items())), by_trigger=dict(by_trig),
               requests=client.requests_made, finished=core.iso(core.utcnow()),
               verdict=("CONTINUE: upper bound clears 150 total and 60 holdout" if len(kept) >= 150 and holdout >= 60
                        else "RETIRE H1: upper bound below the frozen minimum"))
    (OUT / "event_count.json").write_text(json.dumps(log, indent=1))
    with open(OUT / "event_count_sample.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0].keys()) if kept else ["none"])
        w.writeheader()
        for e in kept[:: max(1, len(kept) // 200)]:
            w.writerow(e)
    with open(OUT / "below_cash_proxy_events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(kept[0].keys()) if kept else ["none"])
        w.writeheader()
        w.writerows(proxy)
    print(json.dumps({k: v for k, v in log.items() if k != "notes"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
