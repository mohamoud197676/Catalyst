"""Parse EDGAR payloads and store filings with their acceptance timestamps."""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from ..core import ET, UTC, IngestRun, MarketClock, iso, parse_ts, utcnow
from . import taxonomy


@dataclass(frozen=True)
class FilingRecord:
    accession_no: str
    cik: int
    form_type: str
    filed_date: str | None
    accepted_at: datetime | None
    report_date: str | None
    items: list[str]
    primary_doc: str | None
    size_bytes: int | None


def parse_submissions(payload: dict) -> tuple[dict, list[FilingRecord]]:
    """Turn a submissions API response into an issuer dict and filing records.

    The filing history arrives as parallel arrays ("compact columnar data array" in SEC's
    words). Columns other than accessionNumber and form are treated as optional so a schema
    change at the SEC degrades to missing values instead of a crash or a silent misalignment.
    """
    cik = int(payload["cik"])
    issuer = {
        "cik": cik,
        "name": payload.get("name") or "",
        "entity_type": payload.get("entityType"),
        "sic": payload.get("sic"),
        "sic_description": payload.get("sicDescription"),
        "state_of_inc": payload.get("stateOfIncorporation"),
        "fiscal_year_end": payload.get("fiscalYearEnd"),
        "tickers": list(payload.get("tickers") or []),
        "exchanges": list(payload.get("exchanges") or []),
        "former_names": [
            {"name": f.get("name"), "from": (f.get("from") or "")[:10] or None, "to": (f.get("to") or "")[:10] or None}
            for f in payload.get("formerNames") or []],
        "extra_files": [f.get("name") for f in (payload.get("filings") or {}).get("files") or []],
    }
    recent = (payload.get("filings") or {}).get("recent") or payload   # paged files are bare column sets
    return issuer, parse_filing_columns(cik, recent)


def parse_filing_columns(cik: int, cols: dict) -> list[FilingRecord]:
    accessions = cols.get("accessionNumber") or []
    n = len(accessions)

    def col(name: str) -> list:
        values = cols.get(name)
        if values is None:
            return [None] * n
        if len(values) != n:
            raise ValueError(f"column {name} has {len(values)} values, expected {n}: refusing misaligned data")
        return values

    forms, filed, accepted = col("form"), col("filingDate"), col("acceptanceDateTime")
    report, items, docs, sizes = col("reportDate"), col("items"), col("primaryDocument"), col("size")
    out = []
    for i in range(n):
        if not forms[i]:
            continue
        out.append(FilingRecord(
            accession_no=accessions[i], cik=cik, form_type=forms[i],
            filed_date=filed[i] or None,
            accepted_at=parse_ts(accepted[i]) if accepted[i] else None,
            report_date=report[i] or None,
            items=taxonomy.parse_items(items[i]),
            primary_doc=docs[i] or None,
            size_bytes=int(sizes[i]) if sizes[i] not in (None, "") else None))
    return out


def parse_master_index(text: str) -> list[dict]:
    """Parse an EDGAR master index: pipe-delimited CIK|Company Name|Form Type|Date Filed|Filename.

    Layout confirmed in outline only (see edgar_index in sources.toml). Lines that do not
    fit are skipped and counted by the caller, never guessed at.
    """
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5 or not parts[0].strip().isdigit():
            continue
        m = re.search(r"(\d{10}-\d{2}-\d{6})", parts[4])
        date_filed = parts[3].strip()
        if re.fullmatch(r"\d{8}", date_filed):
            date_filed = f"{date_filed[:4]}-{date_filed[4:6]}-{date_filed[6:]}"
        rows.append({"cik": int(parts[0]), "company": parts[1].strip(), "form_type": parts[2].strip(),
                     "filed_date": date_filed, "accession_no": m.group(1) if m else None,
                     "file_name": parts[4].strip()})
    return rows


def parse_sgml_acceptance(header_text: str) -> datetime | None:
    """Read <ACCEPTANCE-DATETIME>YYYYMMDDHHMMSS from a submission header as Eastern time.

    Used only to cross-check the JSON timestamp (see calibrate_acceptance). The Eastern
    reading of this tag is an assumption until that cross-check has passed on live data.
    """
    m = re.search(r"<ACCEPTANCE-DATETIME>\s*(\d{14})", header_text)
    if not m:
        return None
    return datetime.strptime(m.group(1), "%Y%m%d%H%M%S").replace(tzinfo=ET).astimezone(UTC)


def calibrate_acceptance(json_ts: datetime, header_text: str) -> dict:
    """Compare the two official sources of acceptance time for one filing."""
    sgml = parse_sgml_acceptance(header_text)
    if sgml is None:
        return {"status": "NO_HEADER"}
    diff = abs((json_ts - sgml).total_seconds())
    return {"status": "MATCH" if diff <= 1 else "MISMATCH", "diff_seconds": diff,
            "json_utc": iso(json_ts), "sgml_as_et_utc": iso(sgml)}


def expected_filed_date(accepted_at: datetime, form_type: str, clock: MarketClock) -> str:
    """SEC rule: accepted after 17:30 ET (22:00 for Forms 3, 4, 5) is dated the next business day."""
    local = accepted_at.astimezone(ET)
    base, _ = taxonomy.split_form(form_type)
    cutoff = time(22, 0) if base in taxonomy.OWNERSHIP_FORMS else time(17, 30)
    d = local.date()
    # EDGAR business days are weekdays excluding federal holidays; the exchange calendar is a
    # close stand-in and differs on a few days a year (e.g. Columbus Day, Veterans Day).
    if local.time() > cutoff or local.weekday() >= 5:
        d = d + timedelta(days=1)
        while d.weekday() >= 5:
            d += timedelta(days=1)
    return d.isoformat()


def store_issuer(conn: sqlite3.Connection, issuer: dict, source: str = "edgar_submissions") -> None:
    now = iso(utcnow())
    conn.execute(
        "INSERT INTO issuer (cik, name, entity_type, sic, sic_description, state_of_inc, fiscal_year_end, "
        "first_seen_at_utc, updated_at_utc) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(cik) DO UPDATE SET "
        "name=excluded.name, entity_type=excluded.entity_type, sic=excluded.sic, "
        "sic_description=excluded.sic_description, state_of_inc=excluded.state_of_inc, "
        "fiscal_year_end=excluded.fiscal_year_end, updated_at_utc=excluded.updated_at_utc",
        (issuer["cik"], issuer["name"], issuer["entity_type"], issuer["sic"], issuer["sic_description"],
         issuer["state_of_inc"], issuer["fiscal_year_end"], now, now))
    for f in issuer["former_names"]:
        conn.execute("INSERT OR IGNORE INTO issuer_name_history (cik, name, valid_from, valid_to, source) "
                     "VALUES (?,?,?,?,?)", (issuer["cik"], f["name"], f["from"], f["to"], source))


def store_filings(conn: sqlite3.Connection, records: list[FilingRecord], clock: MarketClock,
                  source: str = "edgar_submissions", run: IngestRun | None = None) -> int:
    """Insert filings not seen before. first_seen_at_utc is set once and never rewritten."""
    new = 0
    now = iso(utcnow())
    for r in records:
        base, amended = taxonomy.split_form(r.form_type)
        event_class, side_hint = taxonomy.classify(r.form_type, r.items)
        session = clock.session_of(r.accepted_at).value if r.accepted_at else None
        cur = conn.execute(
            "INSERT OR IGNORE INTO filing (accession_no, cik, form_type, base_form, is_amendment, filed_date, "
            "accepted_at_utc, accepted_session, report_date, items, primary_doc, size_bytes, event_class, "
            "side_hint, source, first_seen_at_utc) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (r.accession_no, r.cik, r.form_type, base, int(amended), r.filed_date,
             iso(r.accepted_at) if r.accepted_at else None, session, r.report_date,
             ",".join(r.items) or None, r.primary_doc, r.size_bytes, event_class, side_hint, source, now))
        if cur.rowcount:
            new += 1
            for code in r.items:
                label, cls, hint = taxonomy.ITEMS_8K.get(code, (None, None, None))
                conn.execute("INSERT OR IGNORE INTO filing_item VALUES (?,?,?,?,?)",
                             (r.accession_no, code, label, cls, hint))
    if run is not None:
        run.rows_seen += len(records)
        run.rows_new += new
    return new


def derive_listing_events(conn: sqlite3.Connection, cik: int) -> int:
    """Turn end-of-life filings into listing_event rows for every security of the issuer.

    Sources: Form 25 (delisting), Form 15 (deregistration), 8-K item 1.03 (bankruptcy),
    8-K item 3.01 (delisting notice). Event date is the acceptance date. A Form 25 is a
    notice: the actual last trading day comes from price data and is confirmed in Phase 1.
    """
    securities = [r["security_id"] for r in conn.execute("SELECT security_id FROM security WHERE cik=?", (cik,))]
    if not securities:
        return 0
    made = 0
    rows = conn.execute(
        "SELECT f.accession_no, f.event_class, f.accepted_at_utc, f.filed_date, "
        "       (SELECT group_concat(event_class) FROM filing_item i WHERE i.accession_no=f.accession_no) AS item_classes "
        "FROM filing f WHERE f.cik=?", (cik,)).fetchall()
    for row in rows:
        classes = {row["event_class"]} | set((row["item_classes"] or "").split(","))
        event_date = (row["accepted_at_utc"] or row["filed_date"] or "")[:10]
        if not event_date:
            continue
        for cls in classes:
            event_type = taxonomy.LISTING_EVENT_BY_CLASS.get(cls)
            if not event_type:
                continue
            for sid in securities:
                cur = conn.execute(
                    "INSERT OR IGNORE INTO listing_event (security_id, event_type, event_date, reason, source, "
                    "source_ref, first_seen_at_utc) VALUES (?,?,?,?,?,?,?)",
                    (sid, event_type, event_date, cls, "EDGAR_DERIVED", row["accession_no"], iso(utcnow())))
                made += cur.rowcount
    return made


def ingest_cik(conn: sqlite3.Connection, client, cik: int, clock: MarketClock,
               follow_pages: bool = False) -> dict:
    """Fetch one issuer's submissions and store everything new. Network use lives only here."""
    with IngestRun(conn, "edgar_submissions", {"cik": cik, "follow_pages": follow_pages}) as run:
        issuer, records = parse_submissions(client.submissions(cik))
        store_issuer(conn, issuer)
        store_filings(conn, records, clock, run=run)
        if follow_pages:
            for name in issuer["extra_files"]:
                store_filings(conn, parse_filing_columns(cik, client.submissions_page(name)), clock, run=run)
        events = derive_listing_events(conn, cik)
        return {"cik": cik, "seen": run.rows_seen, "new": run.rows_new, "listing_events": events,
                "tickers": issuer["tickers"]}
