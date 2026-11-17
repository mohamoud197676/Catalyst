"""Save one raw IBKR short-stock file, gzip-compressed, under data/borrow/YYYY/MM/DD/.

Raw files are the source of truth. The database can always be rebuilt from them with
`python -m catalyst.cli log-borrow --file <path>`; the reverse is not true. Run this on a
schedule from any always-on machine (see .github/workflows/borrow-logger.yml).
"""
from __future__ import annotations

import gzip
import pathlib
import sys
from datetime import datetime, timezone

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from catalyst.borrow.logger import fetch_ibkr_file, parse_shortstock_file  # noqa: E402


def main(region: str = "usa") -> int:
    now = datetime.now(timezone.utc)
    text = fetch_ibkr_file(region)
    meta, rows = parse_shortstock_file(text)          # refuse to archive a layout we cannot read
    if meta.get("row_count_mismatch"):
        print("truncated download, not archived", file=sys.stderr)
        return 1
    out = pathlib.Path("data/borrow") / now.strftime("%Y/%m/%d") / f"{region}-{now.strftime('%H%M%S')}Z.txt.gz"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(gzip.compress(text.encode("latin-1")))
    print(f"archived {len(rows)} rows to {out} (source stamp {meta['source_asof_raw']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
