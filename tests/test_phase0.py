"""Phase 0 test suite. Runs offline: no test touches the network.

Fixtures marked OBSERVED were read from the live service on 2026-11-16. All others are
synthetic and built to the documented formats.
"""
from __future__ import annotations

import json
import pathlib
import re
import sqlite3
from datetime import date, datetime, timedelta

import pytest

from catalyst import core, costs, paper, prices, secmaster
from catalyst.borrow import logger as bl
from catalyst.core import ET, UTC, Session
from catalyst.edgar import ingest, taxonomy
from catalyst.edgar.client import EdgarClient

SRC = pathlib.Path(core.ROOT, "src")


@pytest.fixture()
def conn():
    c = core.connect(":memory:")
    core.migrate(c)
    return c


@pytest.fixture(scope="session")
def clock():
    return core.MarketClock()


def utc(y, mo, d, h, mi=0, s=0):
    return datetime(y, mo, d, h, mi, s, tzinfo=UTC)


# ============================================================ configuration and guards
def test_project_parameters_are_recorded():
    cfg = core.load_project()
    assert cfg.starting_capital == 2500 and cfg.base_currency == "GBP"
    a = cfg.ambition_summary()
    assert a["multiple"] == 40 and a["months"] == 6
    assert 0.84 < a["required_monthly_return"] < 0.86      # 40x in six months = ~85% a month


def test_non_paper_mode_refuses_to_load(tmp_path):
    for name in ("project.toml", "sources.toml", "costs.toml"):
        (tmp_path / name).write_text((core.CONFIG_DIR / name).read_text())
    p = tmp_path / "project.toml"
    p.write_text(p.read_text().replace('mode = "PAPER"', 'mode = "LIVE"'))
    with pytest.raises(RuntimeError, match="PAPER"):
        core.load_project(tmp_path)


def test_database_refuses_non_paper_order(conn):
    paper.register_strategy(conn, "s1", "LONG", "test")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO paper_order (client_key, mode, strategy_id, book_id, ticker, action, qty, limit_price, "
                     "created_at_utc, status) VALUES ('k','LIVE','s1','LONG','X','BUY',1,1,'t','ACCEPTED')")


def test_database_refuses_short_action_in_long_book(conn):
    paper.register_strategy(conn, "s1", "LONG", "test")
    with pytest.raises(sqlite3.IntegrityError):
        conn.execute("INSERT INTO paper_order (client_key, strategy_id, book_id, ticker, action, qty, limit_price, "
                     "created_at_utc, status) VALUES ('k','s1','LONG','X','SELL_SHORT',1,1,'t','ACCEPTED')")


def test_no_live_execution_code_exists():
    """Fails the build if a broker SDK or an order-placement call is ever added to src/."""
    banned = re.compile(r"\b(ib_insync|ibapi|ib_async|alpaca|placeOrder|place_order|submit_order|tda|robin_stocks)\b")
    hits = [f"{p.name}:{m.group(0)}" for p in SRC.rglob("*.py") for m in banned.finditer(p.read_text())]
    assert hits == []


def test_sizing_and_cost_code_cannot_see_the_target():
    for mod in ("costs.py", "paper.py"):
        text = (SRC / "catalyst" / mod).read_text()
        assert "stretch_target" not in text and "ambition" not in text


def test_registry_blocks_unverified_sources():
    reg = core.SourceRegistry({"a": {"status": "UNVERIFIED", "official_doc_url": "u"},
                               "b": {"status": "PARTIAL"}, "c": {"status": "VERIFIED_DOC"}})
    with pytest.raises(core.SourceNotVerified):
        reg.require("a")
    with pytest.raises(core.SourceNotVerified):
        reg.require("b")
    assert reg.require("b", "PARTIAL") and reg.require("c")
    with pytest.raises(core.SourceNotVerified):
        reg.require("missing")


def test_every_registered_source_has_official_doc_and_date():
    for key, s in core.load_toml("sources.toml").items():
        assert s["official_doc_url"].startswith("https://"), key
        assert s["verified_on"] and s["status"] in core._RANK, key


def test_naive_timestamps_are_refused():
    with pytest.raises(ValueError):
        core.iso(datetime(2026, 1, 1))
    with pytest.raises(ValueError):
        core.parse_ts("2026-01-01T00:00:00")


# ================================================================== market sessions
def test_session_boundaries_winter(clock):
    # 2026-11-12 is a Thursday in EST (UTC-5)
    assert clock.session_of(utc(2026, 11, 12, 8, 59)) is Session.CLOSED     # 03:59 ET
    assert clock.session_of(utc(2026, 11, 12, 9, 0)) is Session.PRE         # 04:00 ET
    assert clock.session_of(utc(2026, 11, 12, 14, 29)) is Session.PRE
    assert clock.session_of(utc(2026, 11, 12, 14, 30)) is Session.REG       # 09:30 ET
    assert clock.session_of(utc(2026, 11, 12, 20, 59)) is Session.REG
    assert clock.session_of(utc(2026, 11, 12, 21, 0)) is Session.POST       # 16:00 ET
    assert clock.session_of(utc(2026, 11, 13, 0, 59)) is Session.POST       # 19:59 ET
    assert clock.session_of(utc(2026, 11, 13, 1, 0)) is Session.CLOSED      # 20:00 ET


def test_session_boundaries_summer_dst(clock):
    # 2026-07-15 is in EDT (UTC-4): open is 13:30 UTC
    assert clock.session_of(utc(2026, 7, 15, 13, 29)) is Session.PRE
    assert clock.session_of(utc(2026, 7, 15, 13, 30)) is Session.REG


def test_weekend_holiday_and_early_close(clock):
    assert clock.session_of(utc(2026, 11, 14, 16, 0)) is Session.CLOSED     # Saturday
    assert clock.session_of(utc(2026, 11, 26, 16, 0)) is Session.CLOSED     # Thanksgiving
    # day after Thanksgiving closes 13:00 ET = 18:00 UTC
    assert clock.session_of(utc(2026, 11, 27, 17, 59)) is Session.REG
    assert clock.session_of(utc(2026, 11, 27, 18, 0)) is Session.POST


def test_next_open_and_close_skip_weekend(clock):
    fri_evening = utc(2026, 11, 13, 23, 0)
    assert clock.next_regular_open(fri_evening) == utc(2026, 11, 16, 14, 30)
    assert clock.next_regular_close(fri_evening) == utc(2026, 11, 16, 21, 0)
    assert clock.nth_close_after(fri_evening, 5) == utc(2026, 11, 20, 21, 0)


# ====================================================================== EDGAR parsing
# OBSERVED: acceptanceDateTime / filingDate pairs read live from
# data.sec.gov/submissions/CIK0000320193.json on 2026-11-16 (first eight rows).
OBSERVED_PAIRS = [
    ("2026-11-12T23:58:20.000Z", "2026-11-12"), ("2026-11-09T22:00:55.000Z", "2026-11-09"),
    ("2026-11-09T21:30:40.000Z", "2026-11-09"), ("2026-11-06T21:30:47.000Z", "2026-11-06"),
    ("2026-11-06T02:54:06.000Z", "2026-11-06"), ("2026-11-05T23:30:13.000Z", "2026-11-05"),
    ("2026-11-05T13:14:16.000Z", "2026-11-05"), ("2026-11-03T13:05:15.000Z", "2026-11-03"),
]


def test_observed_acceptance_times_are_utc_not_eastern():
    """EDGAR accepts filings 06:00-22:00 ET. Read as UTC every observed stamp falls inside that
    window; read as Eastern, 02:54 falls outside it. So the JSON field is true UTC."""
    for stamp, _ in OBSERVED_PAIRS:
        as_utc = core.parse_ts(stamp).astimezone(ET)
        assert 6 <= as_utc.hour < 22, stamp
    naive_hours = [int(s[11:13]) for s, _ in OBSERVED_PAIRS]
    assert any(h < 6 for h in naive_hours)       # the Eastern reading is impossible for at least one


def test_observed_filing_dates_follow_sec_cutoff_rule(clock):
    """Filing date is not acceptance date: 21:54 ET on 5 Nov is dated 6 Nov. Reactions must
    therefore key on acceptance time, never on filing date."""
    stamp, filed = OBSERVED_PAIRS[4]
    accepted = core.parse_ts(stamp)
    assert accepted.astimezone(ET).date().isoformat() == "2026-11-05" and filed == "2026-11-06"
    assert ingest.expected_filed_date(accepted, "8-K", clock) == "2026-11-06"
    # every observed pair is consistent with one of the two cutoff rules
    for s, f in OBSERVED_PAIRS:
        a = core.parse_ts(s)
        assert f in {ingest.expected_filed_date(a, "8-K", clock), ingest.expected_filed_date(a, "4", clock)}


def test_observed_sessions(clock):
    got = [clock.session_of(core.parse_ts(s)).value for s, _ in OBSERVED_PAIRS]
    assert got == ["POST", "POST", "POST", "POST", "CLOSED", "POST", "PRE", "PRE"]


def submissions_fixture():
    """Synthetic issuer in the observed response shape."""
    return {
        "cik": "0009990001", "entityType": "operating", "sic": "2834", "sicDescription": "Pharmaceutical Preparations",
        "name": "Testco Therapeutics Inc", "tickers": ["TSTX"], "exchanges": ["Nasdaq"], "fiscalYearEnd": "1231",
        "stateOfIncorporation": "DE",
        "formerNames": [{"name": "OLDNAME BIO INC", "from": "2019-01-10T05:00:00.000Z", "to": "2023-08-05T04:00:00.000Z"}],
        "filings": {"recent": {
            "accessionNumber": ["0009990001-26-000005", "0009990001-26-000004", "0009990001-26-000003",
                                "0009990001-26-000002", "0009990001-26-000001"],
            "filingDate": ["2026-11-13", "2026-11-12", "2026-11-10", "2026-11-09", "2026-11-06"],
            "reportDate": ["2026-11-12", "2026-11-12", "", "2026-11-09", ""],
            "acceptanceDateTime": ["2026-11-12T22:45:10.000Z", "2026-11-12T12:05:00.000Z", "2026-11-10T21:10:00.000Z",
                                   "2026-11-09T15:00:30.000Z", "2026-11-06T21:01:00.000Z"],
            "form": ["8-K", "8-K", "424B5", "8-K", "S-3/A"],
            "items": ["1.03,9.01", "2.02,9.01", "", "3.01", ""],
            "size": [12000, 250000, 480000, 9000, 310000],
            "primaryDocument": ["a.htm", "b.htm", "c.htm", "d.htm", "e.htm"]},
            "files": [{"name": "CIK0009990001-submissions-001.json"}]}}


def test_parse_submissions_and_classification(clock):
    issuer, recs = ingest.parse_submissions(submissions_fixture())
    assert issuer["cik"] == 9990001 and issuer["tickers"] == ["TSTX"]
    assert issuer["former_names"][0] == {"name": "OLDNAME BIO INC", "from": "2019-01-10", "to": "2023-08-05"}
    assert issuer["extra_files"] == ["CIK0009990001-submissions-001.json"]
    assert len(recs) == 5 and recs[0].items == ["1.03", "9.01"]
    assert recs[0].accepted_at == utc(2026, 11, 12, 22, 45, 10)
    assert taxonomy.classify("8-K", ["1.03", "9.01"]) == ("BANKRUPTCY", "SHORT_WATCH")
    assert taxonomy.classify("8-K", ["2.02", "9.01"]) == ("EARNINGS", "NEUTRAL")
    assert taxonomy.classify("424B5") == ("OFFERING_PROSPECTUS", "SHORT_WATCH")
    assert taxonomy.classify("S-3/A") == ("SHELF_REGISTRATION", "SHORT_WATCH")
    assert taxonomy.classify("UNHEARD-OF") == ("OTHER", "NEUTRAL")
    assert taxonomy.split_form("S-3/A") == ("S-3", True)


def test_misaligned_columns_are_refused():
    bad = submissions_fixture()
    bad["filings"]["recent"]["form"].pop()
    with pytest.raises(ValueError, match="misaligned"):
        ingest.parse_submissions(bad)


def test_missing_optional_columns_degrade_to_null():
    p = submissions_fixture()
    del p["filings"]["recent"]["items"], p["filings"]["recent"]["acceptanceDateTime"]
    _, recs = ingest.parse_submissions(p)
    assert recs[0].items == [] and recs[0].accepted_at is None


def test_store_filings_is_idempotent_and_keeps_first_seen(conn, clock):
    issuer, recs = ingest.parse_submissions(submissions_fixture())
    ingest.store_issuer(conn, issuer)
    assert ingest.store_filings(conn, recs, clock) == 5
    first = conn.execute("SELECT first_seen_at_utc FROM filing WHERE accession_no='0009990001-26-000005'").fetchone()[0]
    conn.execute("UPDATE filing SET first_seen_at_utc='2020-01-01T00:00:00+00:00' WHERE accession_no='0009990001-26-000005'")
    assert ingest.store_filings(conn, recs, clock) == 0          # nothing new on a re-run
    kept = conn.execute("SELECT first_seen_at_utc FROM filing WHERE accession_no='0009990001-26-000005'").fetchone()[0]
    assert kept == "2020-01-01T00:00:00+00:00" != first          # knowledge time is never rewritten
    row = conn.execute("SELECT * FROM filing WHERE accession_no='0009990001-26-000005'").fetchone()
    assert (row["accepted_session"], row["event_class"], row["is_amendment"]) == ("POST", "BANKRUPTCY", 0)
    assert conn.execute("SELECT COUNT(*) FROM filing_item").fetchone()[0] == 5
    assert conn.execute("SELECT name FROM issuer_name_history").fetchone()[0] == "OLDNAME BIO INC"


def test_listing_events_derived_from_filings(conn, clock):
    issuer, recs = ingest.parse_submissions(submissions_fixture())
    ingest.store_issuer(conn, issuer)
    ingest.store_filings(conn, recs, clock)
    sid = secmaster.create_security(conn, cik=9990001)
    assert ingest.derive_listing_events(conn, 9990001) == 2
    events = {(r["event_type"], r["event_date"]) for r in conn.execute("SELECT * FROM listing_event WHERE security_id=?", (sid,))}
    assert events == {("BANKRUPTCY_FILED", "2026-11-12"), ("DELIST_NOTICE", "2026-11-09")}
    assert ingest.derive_listing_events(conn, 9990001) == 0      # idempotent


def test_master_index_parser_skips_non_data_lines():
    text = ("Description: Master Index\nCIK|Company Name|Form Type|Date Filed|Filename\n----\n"
            "9990001|TESTCO THERAPEUTICS INC|8-K|20261112|edgar/data/9990001/0009990001-26-000005.txt\n"
            "9990002|OTHER CO|10-Q|2026-11-12|edgar/data/9990002/0009990002-26-000011.txt\n")
    rows = ingest.parse_master_index(text)
    assert [r["accession_no"] for r in rows] == ["0009990001-26-000005", "0009990002-26-000011"]
    assert rows[0]["filed_date"] == rows[1]["filed_date"] == "2026-11-12"


def test_acceptance_calibration_detects_match_and_mismatch():
    json_ts = utc(2026, 11, 12, 22, 45, 10)                       # 17:45:10 ET
    assert ingest.calibrate_acceptance(json_ts, "<ACCEPTANCE-DATETIME>20261112174510\n")["status"] == "MATCH"
    assert ingest.calibrate_acceptance(json_ts, "<ACCEPTANCE-DATETIME>20261112224510\n")["status"] == "MISMATCH"
    assert ingest.calibrate_acceptance(json_ts, "no header")["status"] == "NO_HEADER"


def test_edgar_client_requires_identity_and_respects_rate_limit():
    with pytest.raises(ValueError):
        EdgarClient("python-requests")
    with pytest.raises(ValueError):
        EdgarClient("M Example m@example.com", max_per_second=20)

    class Resp:
        headers = {}
        def read(self): return b'{"cik":"1","filings":{"recent":{}}}'
        def __enter__(self): return self
        def __exit__(self, *a): return False
    seen = []
    def opener(req, timeout):
        seen.append(req)
        return Resp()
    import time
    c = EdgarClient("M Example m@example.com", max_per_second=10, opener=opener)
    t0 = time.monotonic()
    for _ in range(4):
        c.submissions(1)
    assert time.monotonic() - t0 >= 0.29                          # 4 calls at 10/s take at least 0.3 s
    assert seen[0].get_header("User-agent") == "M Example m@example.com"
    assert seen[0].full_url == "https://data.sec.gov/submissions/CIK0000000001.json"


# =================================================================== security master
def test_ticker_reuse_resolves_point_in_time(conn):
    old = secmaster.create_security(conn, description="Old company")
    new = secmaster.create_security(conn, description="Unrelated new company")
    secmaster.assign_ticker(conn, old, "ABCD", "2015-01-02", "2021-06-30")
    secmaster.assign_ticker(conn, new, "ABCD", "2023-03-01")
    assert secmaster.resolve_ticker(conn, "ABCD", "2019-05-01") == old
    assert secmaster.resolve_ticker(conn, "ABCD", "2022-01-01") is None        # nobody traded as ABCD
    assert secmaster.resolve_ticker(conn, "abcd", "2024-01-01") == new
    with pytest.raises(secmaster.AmbiguousTicker):
        secmaster.assign_ticker(conn, old, "ABCD", "2023-06-01")               # overlap refused


def test_ticker_change_keeps_one_security(conn):
    sid = secmaster.create_security(conn)
    secmaster.assign_ticker(conn, sid, "OLDT", "2020-01-02")
    secmaster.change_ticker(conn, sid, "NEWT", "2024-05-06")
    assert secmaster.resolve_ticker(conn, "OLDT", "2024-05-03") == sid
    assert secmaster.resolve_ticker(conn, "OLDT", "2024-05-06") is None
    assert secmaster.resolve_ticker(conn, "NEWT", "2024-05-06") == sid
    assert secmaster.ticker_on(conn, sid, "2021-01-01") == "OLDT"
    ca = conn.execute("SELECT action_type, old_value, new_value FROM corporate_action").fetchone()
    assert tuple(ca) == ("TICKER_CHANGE", "OLDT", "NEWT")


def test_reverse_split_factor_and_double_reporting(conn):
    sid = secmaster.create_security(conn)
    secmaster.add_corporate_action(conn, sid, "REVERSE_SPLIT", "2025-03-10", split_from=10, split_to=1, source="A")
    secmaster.add_corporate_action(conn, sid, "REVERSE_SPLIT", "2025-03-10", split_from=10, split_to=1, source="B")
    secmaster.add_corporate_action(conn, sid, "SPLIT", "2026-02-02", split_from=1, split_to=2, source="A")
    assert secmaster.price_factor(conn, sid, "2025-03-07", "2025-03-10") == 10.0   # applied once, not twice
    assert secmaster.price_factor(conn, sid, "2025-03-10", "2025-03-11") == 1.0    # ex-date itself is post-split
    assert secmaster.price_factor(conn, sid, "2025-01-01", "2026-06-01") == 5.0
    assert secmaster.price_factor(conn, sid, "2026-06-01", "2025-01-01") == pytest.approx(0.2)
    with pytest.raises(ValueError):
        secmaster.add_corporate_action(conn, sid, "SPLIT", "2026-03-01", split_from=10, split_to=1)


def test_split_not_yet_known_is_invisible_to_a_backtest(conn):
    sid = secmaster.create_security(conn)
    secmaster.add_corporate_action(conn, sid, "REVERSE_SPLIT", "2025-03-10", split_from=10, split_to=1)
    assert secmaster.price_factor(conn, sid, "2025-03-01", "2025-03-20", known_by_utc="2025-03-01T00:00:00+00:00") == 1.0


def test_universe_includes_companies_that_later_died(conn):
    alive = secmaster.create_security(conn); secmaster.assign_ticker(conn, alive, "LIVE", "2018-01-02")
    dead = secmaster.create_security(conn); secmaster.assign_ticker(conn, dead, "DEAD", "2018-01-02")
    bought = secmaster.create_security(conn); secmaster.assign_ticker(conn, bought, "BOUT", "2018-01-02")
    secmaster.record_listing_event(conn, dead, "BANKRUPTCY_FILED", "2023-02-01")
    secmaster.record_listing_event(conn, dead, "DELISTED", "2023-02-15", reason="bankruptcy", last_price=0.04)
    secmaster.record_listing_event(conn, bought, "ACQUIRED", "2022-09-30", last_price=18.5)
    names = lambda d: [r["ticker"] for r in secmaster.universe_asof(conn, d)]
    assert names("2022-06-01") == ["BOUT", "DEAD", "LIVE"]       # all three were tradable then
    assert names("2023-02-10") == ["DEAD", "LIVE"]               # bankrupt but still listed
    assert names("2024-01-01") == ["LIVE"]
    assert secmaster.terminal_event(conn, dead)["last_price"] == 0.04
    assert secmaster.resolve_ticker(conn, "DEAD", "2023-02-15") is None


# ============================================================================ borrow
BORROW_FILE = """#BOF|2026.11.16|09:45:03
#SYM|CUR|NAME|CON|ISIN|REBATERATE|FEERATE|AVAILABLE|
AAAA|USD|ALPHA EASY CO|1001|US0000000011|3.58|0.25|>10000000|
BBBB|USD|BETA HARD CO|1002|US0000000029|-84.12|88.45|1500|
CCCC|USD|GAMMA NONE CO|1003|US0000000037|NA|NA|0|
#EOF|3
"""


def test_borrow_file_parse():
    meta, rows = bl.parse_shortstock_file(BORROW_FILE)
    assert meta["source_asof_raw"] == "2026.11.16 09:45:03" and meta["declared_rows"] == 3
    a, b, c = rows
    assert (a.shortable_shares, a.shortable_is_floor, a.fee_rate_pct) == (10_000_000, True, 0.25)
    assert (b.shortable_shares, b.shortable_is_floor, b.fee_rate_pct, b.rebate_rate_pct) == (1500, False, 88.45, -84.12)
    assert (c.shortable_shares, c.fee_rate_pct) == (0, None)


def test_borrow_parser_is_header_driven_and_refuses_unknown_layouts():
    reordered = "#AVAILABLE|FEERATE|SYM|\n500|12.5|ZZZZ|\n"
    _, rows = bl.parse_shortstock_file(reordered)
    assert (rows[0].ticker, rows[0].shortable_shares, rows[0].fee_rate_pct) == ("ZZZZ", 500, 12.5)
    with pytest.raises(bl.UnknownBorrowLayout):
        bl.parse_shortstock_file("AAAA|USD|X|1|I|1|2|100|\n")
    with pytest.raises(bl.UnknownBorrowLayout):
        bl.parse_shortstock_file("#FOO|BAR|\n")


def test_truncated_borrow_download_is_rejected_and_logged(conn):
    truncated = BORROW_FILE.replace("CCCC|USD|GAMMA NONE CO|1003|US0000000037|NA|NA|0|\n", "")
    with pytest.raises(ValueError, match="truncated"):
        bl.log_snapshot(conn, truncated)
    run = conn.execute("SELECT status, error FROM ingest_run ORDER BY run_id DESC LIMIT 1").fetchone()
    assert run["status"] == "FAILED" and "truncated" in run["error"]
    assert conn.execute("SELECT COUNT(*) FROM borrow_snapshot").fetchone()[0] == 0


def test_borrow_history_is_append_only_and_point_in_time(conn):
    sid = secmaster.create_security(conn); secmaster.assign_ticker(conn, sid, "BBBB", "2020-01-02")
    t1, t2 = utc(2026, 11, 16, 14, 45), utc(2026, 11, 16, 15, 15)
    r1 = bl.log_snapshot(conn, BORROW_FILE, captured_at=t1)
    bl.log_snapshot(conn, BORROW_FILE.replace("88.45|1500", "140.00|0"), captured_at=t2)
    assert r1["rows"] == 3 and r1["unresolved_tickers"] == 2
    assert conn.execute("SELECT COUNT(*) FROM borrow_snapshot").fetchone()[0] == 6
    assert bl.borrow_asof(conn, "BBBB", core.iso(utc(2026, 11, 16, 14, 0))) is None          # before first capture
    mid = bl.borrow_asof(conn, "BBBB", core.iso(utc(2026, 11, 16, 15, 0)))
    assert (mid["fee_rate_pct"], mid["shortable_shares"], mid["security_id"]) == (88.45, 1500, sid)
    late = bl.borrow_asof(conn, "BBBB", core.iso(utc(2026, 11, 16, 16, 0)))
    assert (late["fee_rate_pct"], late["shortable_shares"]) == (140.0, 0)
    assert [h["shortable_shares"] for h in bl.borrow_history(conn, "BBBB")] == [1500, 0]


FTD_FILE = """SETTLEMENT DATE|CUSIP|SYMBOL|QUANTITY (FAILS)|DESCRIPTION|PRICE
20211015|00000A101|ABCD|52000|OLD COMPANY INC|1.42
20241015|99999Z909|ABCD|800|UNRELATED NEW CO|14.10
20241015|11111B202|PENY|125000|PENNY CO|.
Trailer record count 3
"""


def test_ftd_parse_store_and_identifier_map(conn):
    rows = bl.parse_ftd_file(FTD_FILE)
    assert len(rows) == 3 and rows[2]["price"] is None and rows[0]["as_of_date"] == "2021-10-15"
    assert bl.store_ftd(conn, rows) == 3 and bl.store_ftd(conn, rows) == 0
    abcd = [tuple(r)[:4] for r in bl.cusip_symbol_map(conn) if r["ticker"] == "ABCD"]
    # one symbol under two CUSIPs = ticker reuse, exposed by free data
    assert abcd == [("00000A101", "ABCD", "2021-10-15", "2021-10-15"), ("99999Z909", "ABCD", "2024-10-15", "2024-10-15")]


# ================================================================= filing reactions
def minute_bars(start, prices_, volume=1000):
    return [{"t": start + timedelta(minutes=i), "o": p, "h": p, "l": p, "c": p, "v": volume} for i, p in enumerate(prices_)]


def setup_filing(conn, clock, accepted):
    sid = secmaster.create_security(conn); secmaster.assign_ticker(conn, sid, "RXN", "2020-01-02")
    rec = ingest.FilingRecord("0000000042-26-000001", 42, "8-K", None, accepted, None, ["8.01"], None, None)
    ingest.store_filings(conn, [rec], clock)
    return sid


SETTINGS = {"latencies_s": [0, 60], "max_ref_staleness_s": 900, "windows": ["5m", "session_end", "next_open", "next_close"]}


def test_after_hours_reaction_is_measured(conn, clock):
    accepted = utc(2026, 11, 12, 21, 30, 20)                      # 16:30:20 ET, post-market
    sid = setup_filing(conn, clock, accepted)
    # flat 10.00 until the filing, then 11.00 in the minute after, 12.00 thereafter
    prices.store_bars(conn, sid, minute_bars(utc(2026, 11, 12, 21, 20), [10.0] * 10), "1m", "t", clock)
    prices.store_bars(conn, sid, minute_bars(utc(2026, 11, 12, 21, 30), [10.5, 11.0, 12.0, 12.0, 12.0, 12.0, 12.0]), "1m", "t", clock)
    prices.store_bars(conn, sid, minute_bars(utc(2026, 11, 13, 14, 30), [13.0]), "1m", "t", clock)   # next open
    prices.store_bars(conn, sid, [{"t": utc(2026, 11, 13, 14, 30), "o": 13.0, "h": 13.5, "l": 12.5, "c": 12.5, "v": 9e5}], "1d", "t", clock)
    out = {(r["window"], r["latency_s"]): r for r in prices.measure_reaction(conn, "0000000042-26-000001", sid, clock, SETTINGS)}
    r = out[("5m", 0)]
    assert (r["anchor_session"], r["status"], r["ref_price"], r["end_price"]) == ("POST", "OK", 10.0, 12.0)
    assert r["ret"] == pytest.approx(0.20) and r["ref_staleness_s"] == 20 and r["end_session"] == "POST"
    # sixty seconds later the bar holding the news has printed: most of the move is gone
    late = out[("5m", 60)]
    assert late["ref_price"] == 10.5 and late["ret"] == pytest.approx(12.0 / 10.5 - 1)
    assert out[("next_open", 0)]["ret"] == pytest.approx(0.30)
    assert out[("next_close", 0)]["ret"] == pytest.approx(0.25)
    assert conn.execute("SELECT COUNT(*) FROM filing_reaction").fetchone()[0] == 8


def test_premarket_filing_and_untraded_window(conn, clock):
    accepted = utc(2026, 11, 5, 13, 14, 16)                       # 08:14 ET (an observed stamp)
    sid = setup_filing(conn, clock, accepted)
    prices.store_bars(conn, sid, minute_bars(utc(2026, 11, 4, 20, 59), [4.00]), "1m", "t", clock)   # prior close
    res = {(r["window"], r["latency_s"]): r for r in prices.measure_reaction(conn, "0000000042-26-000001", sid, clock, SETTINGS)}
    r = res[("5m", 0)]
    assert r["anchor_session"] == "PRE" and r["status"] == "NO_END_PRICE" and r["n_bars"] == 0
    assert r["ref_price"] == 4.0 and r["ref_staleness_s"] > 900   # stale reference is recorded, not hidden
    assert res[("session_end", 0)]["end_utc"] == core.iso(utc(2026, 11, 5, 14, 30))


def test_filing_while_closed_anchors_to_next_premarket(conn, clock):
    accepted = utc(2026, 11, 6, 2, 54, 6)                         # 21:54 ET, market shut (an observed stamp)
    sid = setup_filing(conn, clock, accepted)
    prices.store_bars(conn, sid, minute_bars(utc(2026, 11, 6, 0, 58), [2.00]), "1m", "t", clock)    # last post-market print
    prices.store_bars(conn, sid, minute_bars(utc(2026, 11, 6, 9, 0), [1.50, 1.40, 1.30, 1.30, 1.20]), "1m", "t", clock)
    r = [x for x in prices.measure_reaction(conn, "0000000042-26-000001", sid, clock, SETTINGS)
         if x["window"] == "5m" and x["latency_s"] == 0][0]
    assert r["anchor_session"] == "CLOSED" and r["end_utc"] == core.iso(utc(2026, 11, 6, 9, 5))
    assert r["end_price"] == 1.20 and r["ret"] == pytest.approx(-0.40) and r["end_session"] == "PRE"


def test_reverse_split_between_reference_and_end_is_not_a_gain(conn, clock):
    accepted = utc(2026, 11, 12, 20, 30)                          # 15:30 ET Thursday
    sid = setup_filing(conn, clock, accepted)
    secmaster.add_corporate_action(conn, sid, "REVERSE_SPLIT", "2026-11-13", split_from=10, split_to=1)
    prices.store_bars(conn, sid, minute_bars(utc(2026, 11, 12, 20, 28), [0.50]), "1m", "t", clock)
    prices.store_bars(conn, sid, [{"t": utc(2026, 11, 13, 14, 30), "o": 4.5, "h": 4.6, "l": 4.0, "c": 4.0, "v": 1e5}], "1d", "t", clock)
    settings = {**SETTINGS, "latencies_s": [0], "windows": ["2d_close"]}
    r = prices.measure_reaction(conn, "0000000042-26-000001", sid, clock, settings)[0]
    # raw 0.50 -> 4.00 looks like +700%; in like units it is 5.00 -> 4.00 = -20%
    assert r["ret"] == pytest.approx(-0.20)


def test_abnormal_return_against_benchmark(conn, clock):
    accepted = utc(2026, 11, 12, 16, 0)
    sid = setup_filing(conn, clock, accepted)
    bench = secmaster.create_security(conn, description="benchmark")
    for s, path in ((sid, [10, 10, 10.6, 10.6, 10.6, 10.6, 10.6]), (bench, [100, 100, 101, 101, 101, 101, 101])):
        prices.store_bars(conn, s, minute_bars(utc(2026, 11, 12, 15, 58), path), "1m", "t", clock)
    settings = {**SETTINGS, "latencies_s": [0], "windows": ["5m"]}
    r = prices.measure_reaction(conn, "0000000042-26-000001", sid, clock, settings, benchmark_security_id=bench)[0]
    assert r["ret"] == pytest.approx(0.06) and r["benchmark_ret"] == pytest.approx(0.01)
    assert r["abnormal_ret"] == pytest.approx(0.05)


def test_inconsistent_bar_is_refused(conn, clock):
    sid = secmaster.create_security(conn)
    with pytest.raises(ValueError):
        prices.store_bars(conn, sid, [{"t": utc(2026, 11, 12, 15, 0), "o": 5, "h": 4, "l": 3, "c": 5, "v": 1}], "1m", "t", clock)


# ======================================================================== cost model
def trade(**kw):
    base = dict(ticker="XYZ", action="BUY", qty=100, price=5.00, on_date="2026-11-16", bid=4.98, ask=5.02,
                window_volume=50_000, daily_volatility=0.06)
    return costs.TradeInputs(**{**base, **kw})


def test_long_buy_cost_breakdown(conn):
    e = costs.estimate(conn, trade())
    assert e.ok and e.participation == pytest.approx(0.002)
    assert e.components["half_spread"] == pytest.approx(2.00)
    assert e.components["commission"] == pytest.approx(1.00)              # 100 x 0.005 = 0.50, minimum 1.00
    assert e.components["impact"] == pytest.approx(1.0 * 0.06 * 0.002 ** 0.5 * 500)
    assert e.components["cat_fee"] == pytest.approx(0.0003)
    assert "sec_fee" not in e.components and "borrow_fee" not in e.components
    assert e.total_usd == pytest.approx(sum(e.components.values()))
    assert e.total_bps == pytest.approx(e.total_usd / 500 * 1e4)
    assert "impact_coefficient" in e.unverified_inputs                    # the assumption is named
    assert conn.execute("SELECT COUNT(*) FROM cost_estimate_log").fetchone()[0] == 1


def test_sell_adds_regulatory_fees_with_dated_schedule(conn):
    e = costs.estimate(conn, trade(action="SELL"))
    assert e.components["sec_fee"] == pytest.approx(500 * 0.0000206)
    assert e.components["finra_taf"] == 0.0                               # fee holiday to 31 Dec 2026
    later = costs.estimate(conn, trade(action="SELL", on_date="2027-02-01"))
    assert later.components["finra_taf"] > 0 and "finra_taf_per_share_sold" in later.unverified_inputs


def test_commission_cap_on_low_priced_shares(conn):
    e = costs.estimate(conn, trade(qty=4000, price=0.25, bid=0.249, ask=0.251, window_volume=2_000_000))
    assert e.components["commission"] == pytest.approx(10.00)             # 4000 x 0.005 = 20, capped at 1% of 1,000
    tiny = costs.estimate(conn, trade(qty=100, price=0.25, bid=0.249, ask=0.251, window_volume=2_000_000))
    assert tiny.components["commission"] == pytest.approx(1.00)           # cap 0.25 < minimum: minimum charged
    assert any("costlier" in n for n in tiny.notes)


def test_tiered_plan_adds_clearing_and_exchange(conn):
    e = costs.estimate(conn, trade(plan="TIERED"))
    assert e.components["commission"] == pytest.approx(0.35) and e.components["clearing"] == pytest.approx(0.02)
    assert "exchange_per_share" in e.unverified_inputs


def test_unpriceable_trades_are_rejected_not_zero_cost(conn):
    assert costs.estimate(conn, trade(window_volume=None)).status == "REJECT_NO_VOLUME"
    assert costs.estimate(conn, trade(qty=1000, window_volume=50_000)).status == "REJECT_PARTICIPATION"   # 2% > 1%
    assert costs.estimate(conn, trade(bid=None, ask=None)).status == "REJECT_NO_QUOTE"
    assert costs.estimate(conn, trade(daily_volatility=None)).status == "REJECT_NO_VOLATILITY"
    est = costs.estimate(conn, trade(bid=None, ask=None, spread_bps_estimate=80))
    assert est.ok and "spread (estimate, no quote)" in est.unverified_inputs
    assert costs.estimate(conn, trade(window_volume=None)).total_usd is None


def short(**kw):
    borrow = {"fee_rate_pct": 36.0, "shortable_shares": 5000, "captured_at_utc": "2026-11-16T14:45:00+00:00"}
    base = dict(action="SELL_SHORT", borrow=borrow, as_of_utc="2026-11-16T15:00:00+00:00", hold_days=5,
                max_fee_rate_pct=50.0, borrow_max_age_minutes=30)
    return trade(**{**base, **kw})


def test_short_borrow_gates(conn):
    assert costs.estimate(conn, short(borrow=None)).status == "REJECT_NO_BORROW_RECORD"
    assert costs.estimate(conn, short(as_of_utc="2026-11-16T16:00:00+00:00")).status == "REJECT_STALE_BORROW"
    assert costs.estimate(conn, short(borrow={"fee_rate_pct": 36.0, "shortable_shares": 50,
                                              "captured_at_utc": "2026-11-16T14:45:00+00:00"})).status == "REJECT_INSUFFICIENT_BORROW"
    assert costs.estimate(conn, short(borrow={"fee_rate_pct": None, "shortable_shares": 5000,
                                              "captured_at_utc": "2026-11-16T14:45:00+00:00"})).status == "REJECT_NO_FEE"
    assert costs.estimate(conn, short(max_fee_rate_pct=25.0)).status == "REJECT_FEE_ABOVE_CAP"
    with pytest.raises(ValueError, match="look-ahead"):
        costs.estimate(conn, short(as_of_utc="2026-11-16T14:00:00+00:00"))


def test_short_carry_cost(conn):
    e = costs.estimate(conn, short())
    assert e.ok
    assert e.components["borrow_fee"] == pytest.approx(500 * 1.02 * 0.36 * 5 / 360)
    assert e.components["sec_fee"] > 0
    assert {"borrow_collateral_markup", "borrow_day_count"} <= set(e.unverified_inputs)


def test_round_trip_and_cost_against_risk_unit(conn):
    rt = costs.round_trip(conn, trade(), exit_price=5.50)
    assert rt["status"] == "OK" and rt["exit"]["action"] == "SELL"
    assert rt["total_usd"] == pytest.approx(rt["entry"]["total_usd"] + rt["exit"]["total_usd"])
    # the fact that drives everything at this account size: on a $500 position the round trip
    # costs more than a third of a 0.5% risk unit on GBP 2,500 (about $16)
    assert rt["total_usd"] > 16 * 0.35
    srt = costs.round_trip(conn, short(), exit_price=4.00)
    assert srt["exit"]["action"] == "BUY_TO_COVER" and "borrow_fee" not in srt["exit"]["components"]
    assert costs.round_trip(conn, short(borrow=None), exit_price=4.0)["status"] == "REJECT_NO_BORROW_RECORD"


# ====================================================================== paper trading
def test_long_and_short_books_are_independent(conn):
    cfg = core.load_project()
    books = {r["book_id"]: r for r in conn.execute("SELECT * FROM book")}
    assert books["LONG"]["starting_equity"] == 1500 and books["SHORT"]["starting_equity"] == 1000
    assert json.loads(books["SHORT"]["limits_json"])["max_fee_rate_pct"] == 25.0
    paper.register_strategy(conn, "long-1", "LONG", "long test")
    paper.register_strategy(conn, "short-1", "SHORT", "short test")
    broker = paper.PaperBroker(conn, cfg)
    with pytest.raises(PermissionError):
        broker.submit(client_key="x", strategy_id="long-1", action="SELL_SHORT", ticker="XYZ", qty=10, limit_price=5, trade=trade())
    filled = broker.submit(client_key="a1", strategy_id="long-1", action="BUY", ticker="XYZ", qty=100, limit_price=5.05, trade=trade())
    assert filled["status"] == "FILLED" and filled["mode"] == "PAPER" and filled["book_id"] == "LONG"
    again = broker.submit(client_key="a1", strategy_id="long-1", action="BUY", ticker="XYZ", qty=100, limit_price=5.05, trade=trade())
    assert again["duplicate"] and again["order_id"] == filled["order_id"]          # retry cannot double an order
    resting = broker.submit(client_key="a2", strategy_id="long-1", action="BUY", ticker="XYZ", qty=100, limit_price=4.90, trade=trade())
    assert resting["status"] == "ACCEPTED"                                         # not marketable: no fill invented
    no_borrow = broker.submit(client_key="b1", strategy_id="short-1", action="SELL_SHORT", ticker="XYZ", qty=100, limit_price=4.90, trade=trade())
    assert (no_borrow["status"], no_borrow["reject_reason"]) == ("REJECTED", "REJECT_NO_BORROW_RECORD")
    pricey = broker.submit(client_key="b2", strategy_id="short-1", action="SELL_SHORT", ticker="XYZ", qty=100, limit_price=4.90, trade=short())
    assert pricey["reject_reason"] == "REJECT_FEE_ABOVE_CAP"                       # 36% fee vs the book's 25% cap
    ok = broker.submit(client_key="b3", strategy_id="short-1", action="SELL_SHORT", ticker="XYZ", qty=100, limit_price=4.90,
                       trade=short(borrow={"fee_rate_pct": 8.0, "shortable_shares": 5000, "captured_at_utc": "2026-11-16T14:45:00+00:00"}))
    assert ok["status"] == "FILLED"
    fill = conn.execute("SELECT * FROM paper_fill WHERE order_id=?", (ok["order_id"],)).fetchone()
    assert fill["price"] == 4.98 and "half_spread" not in json.loads(fill["cost_json"])["charged"]   # spread not double counted
    assert broker.book_summary("LONG") == {"book": "LONG", "orders": 2, "filled": 1, "rejected": 0,
                                           "explicit_costs_usd": pytest.approx(conn.execute(
                                               "SELECT cost_usd FROM paper_fill WHERE order_id=?", (filled["order_id"],)).fetchone()[0])}
    s = broker.book_summary("SHORT")
    assert (s["orders"], s["filled"], s["rejected"]) == (3, 1, 2)


# ============================================================== schema and housekeeping
def test_schema_has_every_documented_table(conn):
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    expected = {"data_source", "ingest_run", "issuer", "issuer_name_history", "security", "ticker_history",
                "listing_event", "corporate_action", "filing", "filing_item", "price_bar", "quote_snapshot",
                "filing_reaction", "fundamental_fact", "borrow_snapshot", "borrow_proxy", "fee_schedule", "cost_estimate_log",
                "book", "strategy", "signal", "paper_order", "paper_fill", "schema_version"}
    assert expected <= tables


def test_migration_is_repeatable_and_wal_on_disk(tmp_path):
    path = tmp_path / "t.db"
    c = core.connect(path); core.migrate(c); core.migrate(c)
    assert c.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert c.execute("SELECT COUNT(*) FROM data_source").fetchone()[0] == len(core.load_toml("sources.toml"))
    assert c.execute("SELECT COUNT(*) FROM fee_schedule").fetchone()[0] == len(core.load_toml("costs.toml")["fee"])


def test_cli_offline_commands(tmp_path, capsys):
    from catalyst import cli
    db = str(tmp_path / "c.db")
    assert cli.main(["--db", db, "init-db"]) == 0
    assert cli.main(["--db", db, "ambition"]) == 0
    f = tmp_path / "usa.txt"; f.write_text(BORROW_FILE)
    assert cli.main(["--db", db, "log-borrow", "--file", str(f)]) == 0
    out = capsys.readouterr().out
    assert "84.9%" in out and '"rows": 3' in out
