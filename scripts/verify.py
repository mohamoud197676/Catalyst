"""Verification pass run on GitHub (re-run 2026-11-19 for new odd-lot hits) (it can reach the SEC directly).

A. Odd-lot offers flagged by the scanner: pull expiry wording, price terms and any later
   tender-offer filings by the same company (amendments, final results).
B. Event-count audit: for a random sample of below-cash proxy events, read the filing header's
   SUBJECT COMPANY and check it matches the company the count assigned. Measures the error
   rate of the subject-company heuristic instead of assuming it.
"""
from __future__ import annotations

import csv
import json
import os
import pathlib
import random
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from catalyst.edgar.client import EdgarClient  # noqa: E402
from catalyst.edgar.oddlot import analyse  # noqa: E402

OUT = pathlib.Path("results/verification")
EXPIRY = re.compile(r"(?:will|shall)\s+expire\s+at\s+[^.;]{0,160}|Expiration\s+(?:Date|Time)[^.;]{0,40}means[^.;]{0,160}", re.I)
SUBJECT = re.compile(r"SUBJECT COMPANY:.*?COMPANY CONFORMED NAME:\s*(.+?)\s{2,}.*?CENTRAL INDEX KEY:\s*(\d+)", re.S)
FILED_BY = re.compile(r"FILED BY:.*?COMPANY CONFORMED NAME:\s*(.+?)\s{2,}", re.S)


def clean(t: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", t))


def part_a(client: EdgarClient) -> list[dict]:
    path = pathlib.Path("results/oddlot/latest_odd_lot.csv")
    rows = list(csv.DictReader(open(path))) if path.exists() else []
    out = []
    for r in rows:
        text = client.get(r["url"])[:6_000_000].decode("latin-1")
        a, c = analyse(text), clean(text)
        expiry = sorted({m.group(0).strip()[:220] for m in EXPIRY.finditer(c)})[:3]
        subs = json.loads(client.get(f"https://data.sec.gov/submissions/CIK{int(r['cik']):010d}.json"))
        rec = subs["filings"]["recent"]
        later = [f"{d} {f} {acc}" for f, d, acc in zip(rec["form"], rec["filingDate"], rec["accessionNumber"])
                 if ("TO-" in f or "13E" in f or f in ("8-K",)) and d >= r["filed"]][:12]
        out.append({"company": r["company"], "tickers": " ".join(subs.get("tickers") or []), "form": r["form"],
                    "filed": r["filed"], "price_terms": " | ".join(a["price_ranges"] or a["prices"]),
                    "expiry_wording": " || ".join(expiry), "odd_lot_snippet": a["snippet"][:400],
                    "later_filings": " ; ".join(later), "url": r["url"]})
    return out


def part_b(client: EdgarClient, n: int = 25, seed: int = 17) -> dict:
    path = pathlib.Path("results/below_cash_proxy_events.csv")
    rows = list(csv.DictReader(open(path)))
    random.Random(seed).shuffle(rows)
    checked = []
    for r in rows[:n]:
        acc = r["accession"]
        url = f"https://www.sec.gov/Archives/edgar/data/{r['cik']}/{acc.replace('-', '')}/{acc}.txt"
        try:
            head = client.get(url)[:20000].decode("latin-1")
        except Exception as exc:  # noqa: BLE001
            checked.append({**r, "header_subject": f"FETCH FAILED {exc!r}"[:120], "match": ""})
            continue
        m, fb = SUBJECT.search(head), FILED_BY.search(head)
        subj_name, subj_cik = (m.group(1).strip(), int(m.group(2))) if m else ("", 0)
        checked.append({"filed": r["filed"], "company_assigned": r["company"], "cik_assigned": r["cik"],
                        "header_subject": subj_name, "header_subject_cik": subj_cik,
                        "filed_by": fb.group(1).strip() if fb else "", "match": subj_cik == int(r["cik"]),
                        "public_float": r["public_float"], "adjusted_net_cash": r["adjusted_net_cash"], "url": url})
    good = [c for c in checked if c.get("match") is True]
    return {"sampled": len(checked), "subject_matches": len(good), "rows": checked}


def main() -> int:
    client = EdgarClient(os.environ.get("SEC_USER_AGENT", ""))
    OUT.mkdir(parents=True, exist_ok=True)
    a = part_a(client)
    with open(OUT / "oddlot_checked.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(a[0].keys()) if a else ["none"])
        w.writeheader(); w.writerows(a)
    b = part_b(client)
    with open(OUT / "subject_audit.csv", "w", newline="") as fh:
        keys = sorted({k for r in b["rows"] for k in r})
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader(); w.writerows(b["rows"])
    print(json.dumps({"oddlot_checked": len(a), "subject_sampled": b["sampled"], "subject_matches": b["subject_matches"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
