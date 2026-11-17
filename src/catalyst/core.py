"""Config, database, clock and source-registry plumbing shared by every module."""
from __future__ import annotations

import enum
import json
import sqlite3
import tomllib
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = ROOT / "config"
SCHEMA_PATH = Path(__file__).with_name("schema.sql")
SCHEMA_VERSION = 1
UTC = timezone.utc
ET = ZoneInfo("America/New_York")


# ----------------------------------------------------------------------------- time
def utcnow() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def iso(ts: datetime) -> str:
    """Canonical storage form: UTC, second precision, explicit offset."""
    if ts.tzinfo is None:
        raise ValueError("naive datetime refused: every timestamp must carry a zone")
    return ts.astimezone(UTC).replace(microsecond=0).isoformat()


def parse_ts(text: str) -> datetime:
    """Parse an ISO timestamp that carries a zone. 'Z' is accepted."""
    ts = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if ts.tzinfo is None:
        raise ValueError(f"timestamp without zone refused: {text!r}")
    return ts.astimezone(UTC)


class Session(str, enum.Enum):
    PRE = "PRE"        # 04:00 ET to the regular open
    REG = "REG"        # regular open to regular close (early closes respected)
    POST = "POST"      # regular close to 20:00 ET
    CLOSED = "CLOSED"  # overnight, weekends, holidays


class MarketClock:
    """US equity sessions. Regular hours and holidays come from exchange_calendars (XNYS).

    Extended hours are taken as 04:00 ET to open and close to 20:00 ET. On early-close days
    real post-market trading ends earlier at some venues; this is a known simplification.
    """

    PRE_OPEN = time(4, 0)
    POST_CLOSE = time(20, 0)

    def __init__(self) -> None:
        import exchange_calendars as xcals

        self._cal = xcals.get_calendar("XNYS")

    def is_trading_day(self, d: date) -> bool:
        return bool(self._cal.is_session(d.isoformat()))

    def regular_bounds(self, d: date) -> tuple[datetime, datetime]:
        s = d.isoformat()
        return (self._cal.session_open(s).to_pydatetime().astimezone(UTC),
                self._cal.session_close(s).to_pydatetime().astimezone(UTC))

    def extended_bounds(self, d: date) -> tuple[datetime, datetime]:
        return (datetime.combine(d, self.PRE_OPEN, ET).astimezone(UTC),
                datetime.combine(d, self.POST_CLOSE, ET).astimezone(UTC))

    def session_of(self, ts: datetime) -> Session:
        d = ts.astimezone(ET).date()
        if not self.is_trading_day(d):
            return Session.CLOSED
        ro, rc = self.regular_bounds(d)
        eo, ec = self.extended_bounds(d)
        if ro <= ts < rc:
            return Session.REG
        if eo <= ts < ro:
            return Session.PRE
        if rc <= ts < ec:
            return Session.POST
        return Session.CLOSED

    def next_trading_day(self, d: date, include_self: bool = False) -> date:
        cur = d if include_self else d + timedelta(days=1)
        for _ in range(15):
            if self.is_trading_day(cur):
                return cur
            cur += timedelta(days=1)
        raise RuntimeError("no trading day found within 15 days")

    def next_regular_open(self, ts: datetime) -> datetime:
        """First regular open strictly after ts."""
        d = ts.astimezone(ET).date()
        if self.is_trading_day(d) and ts < self.regular_bounds(d)[0]:
            return self.regular_bounds(d)[0]
        return self.regular_bounds(self.next_trading_day(d))[0]

    def next_regular_close(self, ts: datetime) -> datetime:
        """First regular close strictly after ts."""
        d = ts.astimezone(ET).date()
        if self.is_trading_day(d) and ts < self.regular_bounds(d)[1]:
            return self.regular_bounds(d)[1]
        return self.regular_bounds(self.next_trading_day(d))[1]

    def session_end(self, ts: datetime) -> datetime | None:
        """End of the session ts falls in; None when the market is closed."""
        d = ts.astimezone(ET).date()
        s = self.session_of(ts)
        if s is Session.PRE:
            return self.regular_bounds(d)[0]
        if s is Session.REG:
            return self.regular_bounds(d)[1]
        if s is Session.POST:
            return self.extended_bounds(d)[1]
        return None

    def nth_close_after(self, ts: datetime, n: int) -> datetime:
        close = self.next_regular_close(ts)
        for _ in range(n - 1):
            close = self.next_regular_close(close)
        return close


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class ProjectConfig:
    raw: dict

    @property
    def starting_capital(self) -> float:
        return float(self.raw["project"]["starting_capital"])

    @property
    def base_currency(self) -> str:
        return self.raw["project"]["base_currency"]

    @property
    def execution_mode(self) -> str:
        return self.raw["execution"]["mode"]

    def book(self, book_id: str) -> dict:
        return dict(self.raw["books"][book_id])

    @property
    def reactions(self) -> dict:
        return dict(self.raw["reactions"])

    def ambition_summary(self) -> dict:
        """Reporting only. Deliberately not imported by sizing or risk code."""
        p = self.raw["project"]
        multiple = p["stretch_target"] / p["starting_capital"]
        monthly = multiple ** (1 / p["ambition_months"]) - 1
        return {"multiple": multiple, "months": p["ambition_months"],
                "required_monthly_return": monthly}


def load_toml(name: str, config_dir: Path | None = None) -> dict:
    with open((config_dir or CONFIG_DIR) / name, "rb") as fh:
        return tomllib.load(fh)


def load_project(config_dir: Path | None = None) -> ProjectConfig:
    cfg = ProjectConfig(load_toml("project.toml", config_dir))
    if cfg.execution_mode != "PAPER":
        raise RuntimeError("execution.mode must be PAPER; no other mode is implemented")
    return cfg


# ----------------------------------------------------------------- source registry
class SourceNotVerified(RuntimeError):
    pass


_RANK = {"BLOCKED": 0, "UNVERIFIED": 1, "PARTIAL": 2, "VERIFIED_DOC": 3, "VERIFIED_LIVE": 4}


class SourceRegistry:
    """Adapters ask permission here before touching the network."""

    def __init__(self, sources: dict | None = None) -> None:
        self.sources = sources if sources is not None else load_toml("sources.toml")

    def status(self, key: str) -> str:
        if key not in self.sources:
            raise SourceNotVerified(f"source {key!r} is not in config/sources.toml")
        return self.sources[key]["status"]

    def require(self, key: str, minimum: str = "VERIFIED_DOC") -> dict:
        status = self.status(key)
        if _RANK[status] < _RANK[minimum]:
            raise SourceNotVerified(
                f"source {key!r} is {status}; {minimum} or better is required. "
                f"Verify it against {self.sources[key].get('official_doc_url')} and update sources.toml.")
        return self.sources[key]


# ------------------------------------------------------------------------- database
def connect(path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path), timeout=30, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if str(path) != ":memory:":
        conn.execute("PRAGMA journal_mode = WAL")   # logger and ingester can write concurrently
        conn.execute("PRAGMA synchronous = NORMAL")
    return conn


def migrate(conn: sqlite3.Connection, config_dir: Path | None = None) -> None:
    conn.executescript(SCHEMA_PATH.read_text())
    conn.execute("INSERT OR IGNORE INTO schema_version VALUES (?, ?)", (SCHEMA_VERSION, iso(utcnow())))
    sync_reference(conn, config_dir)


def sync_reference(conn: sqlite3.Connection, config_dir: Path | None = None) -> None:
    """Mirror sources.toml, costs.toml and book settings into the database."""
    now = iso(utcnow())
    for key, s in load_toml("sources.toml", config_dir).items():
        conn.execute(
            "INSERT INTO data_source VALUES (?,?,?,?,?,?,?) ON CONFLICT(source_key) DO UPDATE SET "
            "name=excluded.name, capability=excluded.capability, official_doc_url=excluded.official_doc_url, "
            "status=excluded.status, verified_on=excluded.verified_on, notes=excluded.notes",
            (key, s["name"], s["capability"], s.get("official_doc_url"), s["status"],
             s.get("verified_on"), s.get("notes")))
    for f in load_toml("costs.toml", config_dir)["fee"]:
        conn.execute(
            "INSERT INTO fee_schedule (broker, plan, component, value, unit, effective_from, effective_to, "
            "verified, note) VALUES (?,?,?,?,?,?,?,?,?) ON CONFLICT(broker, plan, component, effective_from) "
            "DO UPDATE SET value=excluded.value, unit=excluded.unit, effective_to=excluded.effective_to, "
            "verified=excluded.verified, note=excluded.note",
            (f["broker"], f["plan"], f["component"], f["value"], f["unit"], f["effective_from"],
             f.get("effective_to"), int(f["verified"]), f.get("note")))
    cfg = load_project(config_dir)
    for book_id in ("LONG", "SHORT"):
        limits = cfg.book(book_id)
        conn.execute(
            "INSERT INTO book VALUES (?,?,?,?,?) ON CONFLICT(book_id) DO UPDATE SET "
            "starting_equity=excluded.starting_equity, limits_json=excluded.limits_json",
            (book_id, cfg.base_currency, cfg.starting_capital * limits["share_of_capital"],
             json.dumps(limits), now))


class IngestRun:
    """Context manager that records every ingest attempt, including failures."""

    def __init__(self, conn: sqlite3.Connection, source_key: str, params: dict | None = None) -> None:
        self.conn, self.source_key, self.params = conn, source_key, params or {}
        self.rows_seen = self.rows_new = 0
        self.run_id: int | None = None

    def __enter__(self) -> "IngestRun":
        cur = self.conn.execute(
            "INSERT INTO ingest_run (source_key, started_utc, params) VALUES (?,?,?)",
            (self.source_key, iso(utcnow()), json.dumps(self.params)))
        self.run_id = cur.lastrowid
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.conn.execute(
            "UPDATE ingest_run SET finished_utc=?, status=?, rows_seen=?, rows_new=?, error=? WHERE run_id=?",
            (iso(utcnow()), "FAILED" if exc else "OK", self.rows_seen, self.rows_new,
             repr(exc) if exc else None, self.run_id))
        return False
