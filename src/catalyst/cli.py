"""Command line entry point:  python -m catalyst.cli <command>

Commands
  init-db                      create or migrate the database
  sources                      print the capability registry and its verification status
  ambition                     print what the stretch target implies (reporting only)
  ingest-cik CIK [CIK ...]     fetch SEC submissions for issuers          (needs --user-agent)
  log-borrow [--file PATH]     append one borrow snapshot (from IBKR FTP, or a saved file)
  calibrate CIK ACCESSION      cross-check JSON acceptance time against the SGML header
  smoke                        run every live check in order and report pass or fail
"""
from __future__ import annotations

import argparse
import json
import sys

from . import core
from .borrow import logger as borrow_logger
from .edgar import ingest as edgar_ingest
from .edgar.client import EdgarClient


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="catalyst")
    ap.add_argument("--db", default="catalyst.db")
    ap.add_argument("--user-agent", help="'Your Name you@example.com' as SEC asks")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init-db")
    sub.add_parser("sources")
    sub.add_parser("ambition")
    p = sub.add_parser("ingest-cik"); p.add_argument("ciks", nargs="+", type=int); p.add_argument("--all-pages", action="store_true")
    p = sub.add_parser("log-borrow"); p.add_argument("--file"); p.add_argument("--region", default="usa")
    p = sub.add_parser("calibrate"); p.add_argument("cik", type=int); p.add_argument("accession")
    sub.add_parser("smoke")
    sub.add_parser("simulate")
    args = ap.parse_args(argv)

    conn = core.connect(args.db)
    core.migrate(conn)

    if args.cmd == "init-db":
        print(f"database ready at {args.db}, schema v{core.SCHEMA_VERSION}")
    elif args.cmd == "sources":
        for r in conn.execute("SELECT source_key, status, verified_on, capability FROM data_source ORDER BY status, source_key"):
            print(f"{r['status']:<14} {r['source_key']:<22} {r['verified_on']}  {r['capability']}")
    elif args.cmd == "ambition":
        a = core.load_project().ambition_summary()
        print(f"{a['multiple']:.0f}x in {a['months']} months needs {a['required_monthly_return']:.1%} compounded monthly. "
              "Reported only; not used by sizing.")
    elif args.cmd == "ingest-cik":
        client, clock = _client(args), core.MarketClock()
        for cik in args.ciks:
            print(json.dumps(edgar_ingest.ingest_cik(conn, client, cik, clock, follow_pages=args.all_pages)))
    elif args.cmd == "log-borrow":
        text = open(args.file, encoding="latin-1").read() if args.file else borrow_logger.fetch_ibkr_file(args.region)
        print(json.dumps(borrow_logger.log_snapshot(conn, text, args.region)))
    elif args.cmd == "calibrate":
        client = _client(args)
        row = conn.execute("SELECT accepted_at_utc FROM filing WHERE accession_no=?", (args.accession,)).fetchone()
        if row is None:
            print("ingest the issuer first"); return 2
        head = client.filing_text_head(args.cik, args.accession)
        print(json.dumps(edgar_ingest.calibrate_acceptance(core.parse_ts(row["accepted_at_utc"]), head)))
    elif args.cmd == "simulate":
        from .sim import account, validate
        print(json.dumps({"test_of_the_test": validate.run(trials=600), "six_months": account.report(),
                          "markov_ceiling_zero_edge": account.markov_ceiling(0.0)}, indent=1, default=float))
    elif args.cmd == "smoke":
        return _smoke(conn, args)
    return 0


def _client(args) -> EdgarClient:
    if not args.user_agent:
        sys.exit("--user-agent 'Your Name you@example.com' is required for SEC requests")
    return EdgarClient(args.user_agent)


def _smoke(conn, args) -> int:
    """Live checks to run once on a machine with open internet. Each prints PASS or FAIL."""
    results = []

    def check(name, fn):
        try:
            results.append((name, "PASS", fn()))
        except Exception as exc:  # noqa: BLE001 - a smoke test reports everything
            results.append((name, "FAIL", repr(exc)))

    clock = core.MarketClock()
    client = _client(args)
    check("edgar submissions fetch + parse (CIK 320193)",
          lambda: edgar_ingest.ingest_cik(conn, client, 320193, clock))

    def tz_check():
        row = conn.execute("SELECT accession_no, accepted_at_utc FROM filing WHERE cik=320193 AND accepted_at_utc "
                           "IS NOT NULL ORDER BY accepted_at_utc DESC LIMIT 1").fetchone()
        res = edgar_ingest.calibrate_acceptance(core.parse_ts(row["accepted_at_utc"]),
                                                client.filing_text_head(320193, row["accession_no"]))
        if res["status"] != "MATCH":
            raise AssertionError(res)
        return res
    check("acceptance time: JSON (UTC) equals SGML header (Eastern)", tz_check)
    check("ibkr borrow file fetch + parse + append",
          lambda: borrow_logger.log_snapshot(conn, borrow_logger.fetch_ibkr_file("usa")))
    for name, status, detail in results:
        print(f"{status}  {name}\n      {detail}")
    return 0 if all(s == "PASS" for _, s, _ in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
