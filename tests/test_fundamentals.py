"""Fundamentals and the H1 signal. Synthetic fixture in the assumed company-facts layout."""
import pytest

from catalyst import core, fundamentals as f


def fact(end, val, filed, accn, start=None, form="10-Q"):
    d = {"end": end, "val": val, "accn": accn, "form": form, "filed": filed}
    if start:
        d["start"] = start
    return d


PAYLOAD = {"cik": 9990001, "facts": {
    "us-gaap": {
        "CashAndCashEquivalentsAtCarryingValue": {"units": {"USD": [
            fact("2024-06-30", 60e6, "2024-08-08", "a2"), fact("2024-09-30", 50e6, "2024-11-07", "a3")]}},
        "ShortTermInvestments": {"units": {"USD": [fact("2024-09-30", 20e6, "2024-11-07", "a3")]}},
        "LongTermDebt": {"units": {"USD": [fact("2024-09-30", 5e6, "2024-11-07", "a3")]}},
        "LongTermDebtNoncurrent": {"units": {"USD": [fact("2024-09-30", 4e6, "2024-11-07", "a3")]}},
        "ConvertibleNotesPayable": {"units": {"USD": [fact("2024-09-30", 3e6, "2024-11-07", "a3")]}},
        "NetCashProvidedByUsedInOperatingActivities": {"units": {"USD": [
            fact("2024-03-31", -9e6, "2024-05-09", "a1", start="2024-01-01"),
            fact("2024-06-30", -19e6, "2024-08-08", "a2", start="2024-01-01"),
            fact("2024-09-30", -31e6, "2024-11-07", "a3", start="2024-01-01"),
            {"end": "2024-09-30", "accn": "bad"}]}}},          # malformed row: must be skipped and counted
    "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        fact("2024-10-31", 40e6, "2024-11-07", "a3")]}}}}}


@pytest.fixture()
def conn():
    c = core.connect(":memory:"); core.migrate(c)
    assert f.store_companyfacts(c, PAYLOAD) == {"cik": 9990001, "stored": 10, "skipped": 1}
    assert f.store_companyfacts(c, PAYLOAD)["stored"] == 0       # idempotent
    return c


def test_net_cash_uses_only_what_was_public(conn):
    before = f.net_cash_asof(conn, 9990001, "2024-11-06")        # Q3 not filed yet
    assert before["balance_sheet_date"] == "2024-06-30" and before["net_cash"] == 60e6
    after = f.net_cash_asof(conn, 9990001, "2024-11-07")
    # debt = larger of reported total (5m) and sum of parts (4m + 3m): the worse reading
    assert (after["cash"], after["investments"], after["debt"], after["net_cash"]) == (50e6, 20e6, 7e6, 63e6)
    assert f.net_cash_asof(conn, 9990001, "2024-01-01") is None
    assert f.net_cash_asof(conn, 9990001, "2025-09-01") is None  # stale balance sheet is not used


def test_quarterly_burn_from_cumulative_figures(conn):
    q = f.quarterly_operating_cash_flows(conn, 9990001, "2024-12-01")
    assert q == [("2024-09-30", -12e6), ("2024-06-30", -10e6), ("2024-03-31", -9e6)]
    assert f.quarterly_burn_asof(conn, 9990001, "2024-12-01") == 11e6
    assert f.quarterly_burn_asof(conn, 9990001, "2024-08-08") == 9.5e6
    assert f.quarterly_burn_asof(conn, 9990001, "2024-05-10") is None   # one quarter is not enough


def test_h1_signal_matches_the_preregistered_definition(conn):
    # adjusted net cash = 63m - 2 x 11m = 41m; 40m shares
    assert f.h1_signal(conn, 9990001, "2024-11-08", price=1.00)["signal"] is True     # 40m < 41m
    s = f.h1_signal(conn, 9990001, "2024-11-08", price=1.10)                          # 44m > 41m
    assert s["signal"] is False and s["adjusted_net_cash"] == 41e6 and s["market_value"] == pytest.approx(44e6)
    # after a 1-for-10 reverse split the cover-page count is stale by a factor of 10
    assert f.h1_signal(conn, 9990001, "2024-11-08", price=10.0, split_factor_since_cover=10)["signal"] is True
    missing = f.h1_signal(conn, 9990001, "2024-05-10", price=1.0)
    assert missing["signal"] is False and missing["status"] == "MISSING_INPUT" and "burn" in missing["missing"]
