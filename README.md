# catalyst: Phase 0 research foundation

Paper mode only. There is no live-order code in this repository and a test fails if any is added.

## What is here
| Path | Purpose |
| --- | --- |
| `src/catalyst/schema.sql` | Database structure (23 tables), point-in-time and append-only |
| `src/catalyst/core.py` | Config, market sessions, source registry, database plumbing |
| `src/catalyst/secmaster.py` | Ticker reuse and changes, splits, delistings, survivorship-free universe |
| `src/catalyst/edgar/` | SEC client, parsers, form and 8-K item taxonomy, ingester |
| `src/catalyst/prices.py` | Bars with session tags; filing reaction measurement |
| `src/catalyst/borrow/logger.py` | Borrow snapshots and public proxies |
| `src/catalyst/costs.py` | Direction-aware trading cost model |
| `src/catalyst/paper.py` | Separate long and short books; simulated fills |
| `config/` | Project parameters, source verification registry, fee schedule |
| `src/catalyst/fundamentals.py` | Point-in-time cash, debt, burn and the H1 signal |
| `src/catalyst/engine/model.py` | Corporate Move Engine: move model, payoff book, two-move search, Kelly-capped sizing |
| `src/catalyst/backtest/events.py` | Event scorer and the frozen H1 verdict |
| `src/catalyst/sim/` | Test-of-the-test simulation and six-month account simulation |
| `tests/` | 63 offline tests |
| `scripts/`, `.github/workflows/` | Raw borrow archive job (untested) |

## Run it
```
pip install exchange_calendars pytest
python -m pytest -q                                   # 63 tests, offline
python -m catalyst.cli simulate                        # instrument checks and six-month arithmetic
export PYTHONPATH=src
python -m catalyst.cli init-db
python -m catalyst.cli sources                        # verification status of every data source
python -m catalyst.cli --user-agent "Your Name you@example.com" smoke   # LIVE checks: run this first
python -m catalyst.cli --user-agent "Your Name you@example.com" ingest-cik 320193
python -m catalyst.cli log-borrow                     # one borrow snapshot from IBKR's public file
```

`smoke` is the gate for closing Phase 0. It fetches a real SEC submission, cross-checks the
acceptance timestamp against the filing's own header, and fetches and parses a real borrow
file. None of those three could be run from the build environment.

## Rules the code enforces
- A data source is used only if `config/sources.toml` marks it verified to the level the adapter needs.
- Timestamps without a zone are refused.
- Facts are appended, never updated; every row records when this system first saw it.
- A trade the cost model cannot price is rejected, never priced at zero.
- A short with no borrow record is treated as not borrowable.
- The return target is not visible to sizing or cost code.
