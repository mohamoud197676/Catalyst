"""What did below-cash, triggered companies actually do next? Development period only (2012-2019).

No prices are used, and the 2020-2025 holdout is not touched: looking at its outcomes now
would contaminate the one test that decides H1.

For each below-cash proxy event filed before 2020, apply the frozen exclusions that can be
judged from SEC data (SPACs and shells, funds, banks, insurers, China or Taiwan businesses),
then classify the company's first decisive move in the 365 days after the trigger from the
forms it filed. This measures the mechanism H1 relies on: how often a cash-rich, pressured
company actually returns cash.

Classification (first match in time wins):
  TENDER_OFFER     SC 14D9 or SC TO-T: a third party tendered for its shares (usually for cash)
  MERGER_COMPLETED DEFM14A / PREM14A followed by Form 25 or 15: sold, for cash or stock
  BANKRUPTCY      8-K item 1.03
  REVERSE_MERGER  S-4 filed by the company itself (it issues shares to buy a private company)
  DILUTIVE_RAISE  any 424B prospectus or S-1 by the company
  DELISTED        Form 25 or 15 with none of the above
  STATUS_QUO      none of the above
A merger proxy can be for stock as well as cash; those cases are reported separately as
MERGER_UNCLEAR when no Form 25 or 15 followed.
"""
from __future__ import annotations

import collections
import csv
import json
import os
import pathlib
import sys
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from catalyst.edgar.client import EdgarClient  # noqa: E402

OUT = pathlib.Path("results/dev_outcomes")
EXCLUDE_SIC = {6770}                                     # blank checks
EXCLUDE_SIC_RANGES = [(6020, 6099), (6300, 6411), (6720, 6799)]   # banks, insurers, funds and trusts


def excluded(sub: dict) -> str | None:
    try:
        sic = int(sub.get("sic") or 0)
    except ValueError:
        sic = 0
    if sic in EXCLUDE_SIC or any(a <= sic <= b for a, b in EXCLUDE_SIC_RANGES):
        return f"SIC {sic}"
    addr = (sub.get("addresses") or {}).get("business") or {}
    where = f"{addr.get('stateOrCountryDescription') or ''} {addr.get('country') or ''}".upper()
    if "CHINA" in where or "TAIWAN" in where:
        return "China or Taiwan business address"
    if (sub.get("entityType") or "").lower() == "other" and "SHELL" in (sub.get("flags") or "").upper():
        return "shell"
    return None


def filings(client: EdgarClient, sub: dict) -> list[tuple[str, str, str]]:
    cols = [sub["filings"]["recent"]]
    for f in sub["filings"].get("files") or []:
        cols.append(client.submissions_page(f["name"]))
    out = []
    for c in cols:
        for form, d, items in zip(c["form"], c["filingDate"], c.get("items") or [""] * len(c["form"])):
            out.append((d, form.upper(), items or ""))
    return sorted(out)


def classify(rows: list[tuple[str, str, str]], start: str) -> tuple[str, str]:
    end = (date.fromisoformat(start) + timedelta(days=365)).isoformat()
    w = [r for r in rows if start < r[0] <= end]
    merger_proxy = next((r for r in w if r[1] in ("DEFM14A", "PREM14A")), None)
    for d, form, items in w:
        if form in ("SC 14D9", "SC TO-T", "SCHEDULE 14D-9", "SCHEDULE TO-T"):
            return "TENDER_OFFER", f"{d} {form}"
        if form.startswith("8-K") and "1.03" in items:
            return "BANKRUPTCY", f"{d} 8-K 1.03"
        if form == "S-4":
            return "REVERSE_MERGER", f"{d} S-4"
        if merger_proxy and form in ("25", "25-NSE", "15-12B", "15-12G", "15-15D") and d >= merger_proxy[0]:
            return "MERGER_COMPLETED", f"{merger_proxy[0]} {merger_proxy[1]} then {d} {form}"
        if form.startswith("424B") or form == "S-1":
            return "DILUTIVE_RAISE", f"{d} {form}"
        if form in ("25", "25-NSE", "15-12B", "15-12G", "15-15D"):
            return "DELISTED", f"{d} {form}"
    if merger_proxy:
        return "MERGER_UNCLEAR", f"{merger_proxy[0]} {merger_proxy[1]}"
    return "STATUS_QUO", ""


def main() -> int:
    client = EdgarClient(os.environ.get("SEC_USER_AGENT", ""))
    OUT.mkdir(parents=True, exist_ok=True)
    events = [r for r in csv.DictReader(open("results/below_cash_proxy_events.csv")) if r["filed"] < "2020-01-01"]
    rows, cache = [], {}
    for e in events:
        cik = int(e["cik"])
        if cik not in cache:
            sub = json.loads(client.get(f"https://data.sec.gov/submissions/CIK{cik:010d}.json"))
            cache[cik] = (sub, filings(client, sub))
        sub, fl = cache[cik]
        why = excluded(sub)
        move, evidence = ("EXCLUDED", why) if why else classify(fl, e["filed"])
        rows.append({"filed": e["filed"], "company": e["company"], "cik": cik, "sic": sub.get("sic"),
                     "public_float_m": round(float(e["public_float"]) / 1e6, 2),
                     "adjusted_net_cash_m": round(float(e["adjusted_net_cash"]) / 1e6, 2),
                     "move": move, "evidence": evidence})
    with open(OUT / "events.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    kept = [r for r in rows if r["move"] != "EXCLUDED"]
    big = [r for r in kept if r["public_float_m"] >= 20]
    summary = {"dev_events": len(rows), "excluded": len(rows) - len(kept), "kept": len(kept),
               "moves_all": dict(collections.Counter(r["move"] for r in kept).most_common()),
               "kept_float_20m_plus": len(big),
               "moves_float_20m_plus": dict(collections.Counter(r["move"] for r in big).most_common()),
               "exclusion_reasons": dict(collections.Counter(r["evidence"] for r in rows if r["move"] == "EXCLUDED").most_common(8))}
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
