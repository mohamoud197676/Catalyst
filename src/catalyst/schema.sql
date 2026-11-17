-- Catalyst research store, schema v1.
-- Conventions
--   * Every timestamp column ending _utc is ISO-8601 UTC text ("2026-11-12T23:58:20+00:00").
--   * Two clocks are kept apart on purpose:
--       event time      = when the thing happened in the world (accepted_at_utc, ex_date, bar_start_utc)
--       knowledge time  = when this system could first have known it (first_seen_at_utc, captured_at_utc)
--     Backtests may only use rows whose knowledge time is <= the simulated clock.
--   * Fact tables are append-only. Corrections are new rows, never updates.
--   * Prices are stored UNADJUSTED. Adjustment is computed from corporate_action at query time.

CREATE TABLE IF NOT EXISTS schema_version (
    version      INTEGER PRIMARY KEY,
    applied_at   TEXT NOT NULL
);

-- ------------------------------------------------------------------ provenance
CREATE TABLE IF NOT EXISTS data_source (
    source_key        TEXT PRIMARY KEY,
    name              TEXT NOT NULL,
    capability        TEXT NOT NULL,
    official_doc_url  TEXT,
    status            TEXT NOT NULL CHECK (status IN ('VERIFIED_LIVE','VERIFIED_DOC','PARTIAL','UNVERIFIED','BLOCKED')),
    verified_on       TEXT,
    notes             TEXT
);

CREATE TABLE IF NOT EXISTS ingest_run (
    run_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    source_key    TEXT NOT NULL,
    started_utc   TEXT NOT NULL,
    finished_utc  TEXT,
    status        TEXT NOT NULL DEFAULT 'RUNNING' CHECK (status IN ('RUNNING','OK','FAILED')),
    rows_seen     INTEGER DEFAULT 0,
    rows_new      INTEGER DEFAULT 0,
    params        TEXT,
    error         TEXT
);

-- ------------------------------------------------------------- security master
CREATE TABLE IF NOT EXISTS issuer (
    cik               INTEGER PRIMARY KEY,
    name              TEXT NOT NULL,
    entity_type       TEXT,
    sic               TEXT,
    sic_description   TEXT,
    state_of_inc      TEXT,
    fiscal_year_end   TEXT,
    first_seen_at_utc TEXT NOT NULL,
    updated_at_utc    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS issuer_name_history (
    cik          INTEGER NOT NULL REFERENCES issuer(cik),
    name         TEXT NOT NULL,
    valid_from   TEXT,
    valid_to     TEXT,
    source       TEXT NOT NULL,
    PRIMARY KEY (cik, name, valid_from)
);

-- A security is the thing that trades. It outlives its tickers: tickers are reused by
-- unrelated companies, so nothing downstream may key on ticker.
CREATE TABLE IF NOT EXISTS security (
    security_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    cik               INTEGER REFERENCES issuer(cik),
    cusip             TEXT,
    isin              TEXT,
    figi              TEXT,
    security_type     TEXT NOT NULL DEFAULT 'CS',
    description       TEXT,
    created_at_utc    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_security_cik ON security(cik);

CREATE TABLE IF NOT EXISTS ticker_history (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id       INTEGER NOT NULL REFERENCES security(security_id),
    ticker            TEXT NOT NULL,
    exchange          TEXT,
    valid_from        TEXT NOT NULL,          -- inclusive date
    valid_to          TEXT,                   -- exclusive date; NULL = still current
    source            TEXT NOT NULL,
    first_seen_at_utc TEXT NOT NULL,
    UNIQUE (security_id, ticker, valid_from)
);
CREATE INDEX IF NOT EXISTS ix_ticker_history_ticker ON ticker_history(ticker, valid_from);

-- Life events of a listing: the data that survivorship bias deletes.
CREATE TABLE IF NOT EXISTS listing_event (
    id                    INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id           INTEGER NOT NULL REFERENCES security(security_id),
    event_type            TEXT NOT NULL CHECK (event_type IN (
                              'LISTED','DELIST_NOTICE','DELISTED','MOVED_TO_OTC','DEREGISTERED',
                              'BANKRUPTCY_FILED','BANKRUPTCY_EMERGED','ACQUIRED','MERGED',
                              'TRADING_HALT','TRADING_RESUMED')),
    event_date            TEXT NOT NULL,
    reason                TEXT,
    last_price            REAL,
    successor_security_id INTEGER REFERENCES security(security_id),
    source                TEXT NOT NULL,
    source_ref            TEXT,
    first_seen_at_utc     TEXT NOT NULL,
    UNIQUE (security_id, event_type, event_date, source)
);

CREATE TABLE IF NOT EXISTS corporate_action (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id       INTEGER NOT NULL REFERENCES security(security_id),
    action_type       TEXT NOT NULL CHECK (action_type IN (
                          'SPLIT','REVERSE_SPLIT','STOCK_DIVIDEND','CASH_DIVIDEND','SPINOFF',
                          'TICKER_CHANGE','NAME_CHANGE','CUSIP_CHANGE')),
    ex_date           TEXT NOT NULL,
    announced_at_utc  TEXT,
    split_from        REAL,     -- shares held before
    split_to          REAL,     -- shares held after  (1-for-10 reverse split: from=10, to=1)
    cash_amount       REAL,
    old_value         TEXT,
    new_value         TEXT,
    source            TEXT NOT NULL,
    source_ref        TEXT,
    first_seen_at_utc TEXT NOT NULL,
    UNIQUE (security_id, action_type, ex_date, source),
    CHECK (action_type NOT IN ('SPLIT','REVERSE_SPLIT') OR (split_from > 0 AND split_to > 0))
);

-- --------------------------------------------------------------------- filings
CREATE TABLE IF NOT EXISTS filing (
    accession_no        TEXT PRIMARY KEY,
    cik                 INTEGER NOT NULL,
    form_type           TEXT NOT NULL,
    base_form           TEXT NOT NULL,
    is_amendment        INTEGER NOT NULL DEFAULT 0,
    filed_date          TEXT,                 -- SEC "filing date"; can be the business day AFTER acceptance
    accepted_at_utc     TEXT,                 -- the timestamp reactions are measured from
    accepted_session    TEXT CHECK (accepted_session IN ('PRE','REG','POST','CLOSED')),
    report_date         TEXT,
    items               TEXT,                 -- 8-K item codes, comma separated
    primary_doc         TEXT,
    size_bytes          INTEGER,
    event_class         TEXT,
    side_hint           TEXT CHECK (side_hint IN ('LONG_WATCH','SHORT_WATCH','NEUTRAL')),
    source              TEXT NOT NULL,
    first_seen_at_utc   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_filing_cik_time ON filing(cik, accepted_at_utc);
CREATE INDEX IF NOT EXISTS ix_filing_class ON filing(event_class, accepted_at_utc);

CREATE TABLE IF NOT EXISTS filing_item (
    accession_no  TEXT NOT NULL REFERENCES filing(accession_no),
    item_code     TEXT NOT NULL,
    item_label    TEXT,
    event_class   TEXT,
    side_hint     TEXT,
    PRIMARY KEY (accession_no, item_code)
);

-- ---------------------------------------------------------------------- prices
CREATE TABLE IF NOT EXISTS price_bar (
    security_id     INTEGER NOT NULL REFERENCES security(security_id),
    timespan        TEXT NOT NULL CHECK (timespan IN ('1m','5m','1d')),
    bar_start_utc   TEXT NOT NULL,
    session         TEXT NOT NULL CHECK (session IN ('PRE','REG','POST','CLOSED','DAY')),
    open            REAL NOT NULL,
    high            REAL NOT NULL,
    low             REAL NOT NULL,
    close           REAL NOT NULL,
    volume          REAL NOT NULL,
    vwap            REAL,
    trades          INTEGER,
    source          TEXT NOT NULL,
    ingested_at_utc TEXT NOT NULL,
    PRIMARY KEY (security_id, timespan, bar_start_utc, source)
);

CREATE TABLE IF NOT EXISTS quote_snapshot (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    security_id     INTEGER NOT NULL REFERENCES security(security_id),
    ts_utc          TEXT NOT NULL,
    session         TEXT,
    bid             REAL,
    ask             REAL,
    bid_size        REAL,
    ask_size        REAL,
    source          TEXT NOT NULL,
    captured_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_quote_sec_ts ON quote_snapshot(security_id, ts_utc);

-- One row per filing x security x window x assumed reaction latency.
CREATE TABLE IF NOT EXISTS filing_reaction (
    accession_no      TEXT NOT NULL REFERENCES filing(accession_no),
    security_id       INTEGER NOT NULL REFERENCES security(security_id),
    window_key        TEXT NOT NULL,
    latency_s         INTEGER NOT NULL,
    method_version    TEXT NOT NULL,
    anchor_utc        TEXT NOT NULL,
    anchor_session    TEXT NOT NULL,
    status            TEXT NOT NULL CHECK (status IN ('OK','NO_REF_PRICE','STALE_REF','NO_END_PRICE')),
    ref_price         REAL,
    ref_bar_utc       TEXT,
    ref_staleness_s   INTEGER,
    end_utc           TEXT,
    end_price         REAL,
    end_session       TEXT,
    ret               REAL,
    benchmark_ret     REAL,
    abnormal_ret      REAL,
    volume_in_window  REAL,
    n_bars            INTEGER,
    computed_at_utc   TEXT NOT NULL,
    PRIMARY KEY (accession_no, security_id, window_key, latency_s, method_version)
);

-- ---------------------------------------------------------------------- borrow
CREATE TABLE IF NOT EXISTS borrow_snapshot (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           INTEGER REFERENCES ingest_run(run_id),
    source_key       TEXT NOT NULL,
    region           TEXT NOT NULL,
    source_asof_raw  TEXT,                 -- timestamp printed in the source file, kept verbatim
    captured_at_utc  TEXT NOT NULL,        -- our clock; the only one trusted for point-in-time use
    ticker           TEXT NOT NULL,
    security_id      INTEGER REFERENCES security(security_id),
    currency         TEXT,
    name             TEXT,
    conid            TEXT,
    isin             TEXT,
    fee_rate_pct     REAL,                 -- annualised percent
    rebate_rate_pct  REAL,
    shortable_shares INTEGER,
    shortable_is_floor INTEGER NOT NULL DEFAULT 0,   -- 1 when source printed ">N"
    raw_available    TEXT
);
CREATE INDEX IF NOT EXISTS ix_borrow_ticker_time ON borrow_snapshot(ticker, captured_at_utc);
CREATE INDEX IF NOT EXISTS ix_borrow_sec_time ON borrow_snapshot(security_id, captured_at_utc);

-- Free public series that stand in for borrow tightness before our own log begins.
-- They are proxies. None of them is a fee or a locate.
CREATE TABLE IF NOT EXISTS borrow_proxy (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    proxy_type      TEXT NOT NULL CHECK (proxy_type IN ('FTD','SHORT_VOLUME','SHORT_INTEREST','THRESHOLD_LIST')),
    as_of_date      TEXT NOT NULL,
    ticker          TEXT NOT NULL,
    cusip           TEXT,
    security_id     INTEGER REFERENCES security(security_id),
    quantity        REAL,
    total_volume    REAL,
    price           REAL,
    description     TEXT,
    source_key      TEXT NOT NULL,
    ingested_at_utc TEXT NOT NULL,
    UNIQUE (proxy_type, as_of_date, ticker, cusip, source_key)
);

-- ----------------------------------------------------------------------- costs
CREATE TABLE IF NOT EXISTS fee_schedule (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    broker         TEXT NOT NULL,
    plan           TEXT NOT NULL,
    component      TEXT NOT NULL,
    value          REAL NOT NULL,
    unit           TEXT NOT NULL,
    effective_from TEXT NOT NULL,
    effective_to   TEXT,
    verified       INTEGER NOT NULL DEFAULT 0,
    source_url     TEXT,
    verified_on    TEXT,
    note           TEXT,
    UNIQUE (broker, plan, component, effective_from)
);

CREATE TABLE IF NOT EXISTS cost_estimate_log (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at_utc    TEXT NOT NULL,
    ticker            TEXT NOT NULL,
    action            TEXT NOT NULL,
    qty               REAL NOT NULL,
    price             REAL NOT NULL,
    status            TEXT NOT NULL,
    total_usd         REAL,
    total_bps         REAL,
    breakdown_json    TEXT NOT NULL,
    unverified_inputs TEXT
);

-- ------------------------------------------------- strategies, books, paper trading
-- Long and short are separate books with separate limits and separate results.
CREATE TABLE IF NOT EXISTS book (
    book_id          TEXT PRIMARY KEY CHECK (book_id IN ('LONG','SHORT')),
    base_currency    TEXT NOT NULL,
    starting_equity  REAL NOT NULL,
    limits_json      TEXT NOT NULL,
    created_at_utc   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy (
    strategy_id     TEXT PRIMARY KEY,
    book_id         TEXT NOT NULL REFERENCES book(book_id),
    name            TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'RESEARCH' CHECK (status IN ('RESEARCH','PAPER','RETIRED')),
    prereg_ref      TEXT,
    config_json     TEXT,
    created_at_utc  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS signal (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy_id      TEXT NOT NULL REFERENCES strategy(strategy_id),
    security_id      INTEGER NOT NULL REFERENCES security(security_id),
    generated_at_utc TEXT NOT NULL,
    score            REAL,
    features_json    TEXT,
    gate_json        TEXT,
    status           TEXT NOT NULL DEFAULT 'NEW'
);

-- mode is pinned to PAPER by the database itself, not only by application code.
CREATE TABLE IF NOT EXISTS paper_order (
    order_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    client_key      TEXT NOT NULL UNIQUE,          -- idempotency key: a retry cannot duplicate an order
    mode            TEXT NOT NULL DEFAULT 'PAPER' CHECK (mode = 'PAPER'),
    strategy_id     TEXT NOT NULL REFERENCES strategy(strategy_id),
    book_id         TEXT NOT NULL REFERENCES book(book_id),
    security_id     INTEGER REFERENCES security(security_id),
    ticker          TEXT NOT NULL,
    action          TEXT NOT NULL CHECK (action IN ('BUY','SELL','SELL_SHORT','BUY_TO_COVER')),
    qty             REAL NOT NULL CHECK (qty > 0),
    limit_price     REAL NOT NULL CHECK (limit_price > 0),
    created_at_utc  TEXT NOT NULL,
    status          TEXT NOT NULL CHECK (status IN ('ACCEPTED','REJECTED','FILLED','CANCELLED')),
    reject_reason   TEXT,
    CHECK ((book_id = 'LONG'  AND action IN ('BUY','SELL'))
        OR (book_id = 'SHORT' AND action IN ('SELL_SHORT','BUY_TO_COVER')))
);

CREATE TABLE IF NOT EXISTS paper_fill (
    fill_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id        INTEGER NOT NULL REFERENCES paper_order(order_id),
    filled_at_utc   TEXT NOT NULL,
    qty             REAL NOT NULL,
    price           REAL NOT NULL,
    cost_usd        REAL NOT NULL,
    cost_json       TEXT NOT NULL
);

-- ------------------------------------------------------------------------ views
CREATE VIEW IF NOT EXISTS v_borrow_latest AS
SELECT b.* FROM borrow_snapshot b
JOIN (SELECT ticker, MAX(captured_at_utc) AS m FROM borrow_snapshot GROUP BY ticker) x
  ON x.ticker = b.ticker AND x.m = b.captured_at_utc;

CREATE VIEW IF NOT EXISTS v_terminal_event AS
SELECT security_id, MIN(event_date) AS terminal_date
FROM listing_event
WHERE event_type IN ('DELISTED','ACQUIRED','MERGED','DEREGISTERED')
GROUP BY security_id;

-- ----------------------------------------------------------------- fundamentals
-- One row per reported XBRL fact. filed_date is the knowledge time: a balance sheet for
-- 30 September is unusable until the day its 10-Q was filed.
CREATE TABLE IF NOT EXISTS fundamental_fact (
    cik           INTEGER NOT NULL,
    taxonomy      TEXT NOT NULL,
    tag           TEXT NOT NULL,
    unit          TEXT NOT NULL,
    period_start  TEXT NOT NULL DEFAULT '',     -- '' for point-in-time (balance sheet) facts
    period_end    TEXT NOT NULL,
    value         REAL NOT NULL,
    filed_date    TEXT NOT NULL,
    form          TEXT,
    accession_no  TEXT NOT NULL,
    PRIMARY KEY (cik, taxonomy, tag, unit, period_start, period_end, accession_no)
);
CREATE INDEX IF NOT EXISTS ix_fact_lookup ON fundamental_fact(cik, tag, filed_date);
