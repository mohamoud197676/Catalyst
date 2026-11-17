"""Daily odd-lot tender scan over recent SEC daily indexes. Writes results/oddlot/latest.csv.

First run looks back LOOKBACK days; later runs keep every row ever found (append, de-duplicated).
"""
from __future__ import annotations

import csv
import os
import pathlib
import sys
from datetime import date, timedelta

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from catalyst.edgar.client import EdgarClient  # noqa: E402
from catalyst.edgar.ingest import parse_master_index  # noqa: E402
from catalyst.edgar.oddlot import TENDER_FORMS, analyse  # noqa: E402

OUT = pathlib.Path("results/oddlot")
FIELDS = ["filed", "form", "company", "cik", "accession", "odd_lot_priority", "odd_lot_mentions",
          "price_ranges", "prices", "snippet", "url"]


def main(lookback: int = 45) -> int:
    client = EdgarClient(os.environ.get("SEC_USER_AGENT", ""))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "all.csv"
    seen = {}
    if path.exists():
        with open(path) as fh:
            for r in csv.DictReader(fh):
                seen[r["accession"]] = r
    today = date.today()
    found = 0
    for back in range(lookback, -1, -1):
        d = today - timedelta(days=back)
        if d.weekday() >= 5:
            continue
        try:
            rows = parse_master_index(client.daily_master_index(d.year, (d.month - 1) // 3 + 1, d.strftime("%Y%m%d")))
        except Exception:      # holidays and not-yet-published days have no index
            continue
        for r in rows:
            if r["form_type"].upper() not in TENDER_FORMS or not r["accession_no"] or r["accession_no"] in seen:
                continue
            url = f"https://www.sec.gov/Archives/{r['file_name']}"
            try:
                text = client.get(url)[:6_000_000].decode("latin-1")
            except Exception as exc:  # noqa: BLE001
                print(f"skip {url}: {exc!r}")
                continue
            a = analyse(text)
            seen[r["accession_no"]] = {"filed": r["filed_date"], "form": r["form_type"], "company": r["company"],
                                       "cik": r["cik"], "accession": r["accession_no"],
                                       "odd_lot_priority": a["odd_lot_priority"], "odd_lot_mentions": a["odd_lot_mentions"],
                                       "price_ranges": " | ".join(a["price_ranges"]), "prices": " | ".join(a["prices"]),
                                       "snippet": a["snippet"], "url": url}
            found += 1
    rows = sorted(seen.values(), key=lambda r: r["filed"], reverse=True)
    for name, subset in (("all.csv", rows), ("latest_odd_lot.csv", [r for r in rows if str(r["odd_lot_priority"]) == "True"])):
        with open(OUT / name, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(subset)
    print(f"new filings read: {found}; tender filings on record: {len(rows)}; "
          f"with odd-lot priority: {sum(str(r['odd_lot_priority']) == 'True' for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*(int(a) for a in sys.argv[1:])))
