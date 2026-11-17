"""Thin, polite HTTP client for SEC EDGAR.

Rules taken from SEC's own pages (see config/sources.toml):
  * no more than 10 requests per second per user; this client holds itself to 5
  * a declared User-Agent identifying the requester
No API key exists or is needed.
"""
from __future__ import annotations

import gzip
import json
import threading
import time
import urllib.error
import urllib.request

from ..core import SourceRegistry

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
SUBMISSIONS_PAGE_URL = "https://data.sec.gov/submissions/{name}"
DAILY_INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{qtr}/master.{ymd}.idx"
FILING_TXT_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}.txt"
BULK_SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"


class EdgarClient:
    def __init__(self, user_agent: str, registry: SourceRegistry | None = None,
                 max_per_second: float = 5.0, retries: int = 3, opener=None) -> None:
        if "@" not in user_agent or len(user_agent.split()) < 2:
            raise ValueError("SEC asks for a User-Agent naming the requester and a contact email, "
                             "e.g. 'Jane Smith jane@example.com'")
        if max_per_second > 10:
            raise ValueError("SEC limit is 10 requests per second")
        self.user_agent = user_agent
        self.registry = registry or SourceRegistry()
        self._min_gap = 1.0 / max_per_second
        self._last = 0.0
        self._lock = threading.Lock()
        self.retries = retries
        self._open = opener or urllib.request.urlopen
        self.requests_made = 0

    def _throttle(self) -> None:
        with self._lock:
            wait = self._min_gap - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
            self._last = time.monotonic()

    def get(self, url: str) -> bytes:
        last_err: Exception | None = None
        for attempt in range(self.retries):
            self._throttle()
            req = urllib.request.Request(url, headers={
                "User-Agent": self.user_agent, "Accept-Encoding": "gzip, deflate"})
            try:
                with self._open(req, timeout=30) as resp:
                    self.requests_made += 1
                    body = resp.read()
                    if resp.headers.get("Content-Encoding") == "gzip":
                        body = gzip.decompress(body)
                    return body
            except urllib.error.HTTPError as err:
                last_err = err
                if err.code in (403, 429):      # blocked or rate limited: back off hard, do not hammer
                    time.sleep(10 * (attempt + 1))
                elif err.code >= 500:
                    time.sleep(2 * (attempt + 1))
                else:
                    raise
            except urllib.error.URLError as err:
                last_err = err
                time.sleep(2 * (attempt + 1))
        raise RuntimeError(f"EDGAR request failed after {self.retries} attempts: {url}") from last_err

    def submissions(self, cik: int) -> dict:
        self.registry.require("edgar_submissions")
        return json.loads(self.get(SUBMISSIONS_URL.format(cik=cik)))

    def submissions_page(self, name: str) -> dict:
        """Older filings are split into extra files named in submissions['filings']['files']."""
        self.registry.require("edgar_submissions")
        return json.loads(self.get(SUBMISSIONS_PAGE_URL.format(name=name)))

    def daily_master_index(self, year: int, qtr: int, ymd: str) -> str:
        self.registry.require("edgar_index", "PARTIAL")
        return self.get(DAILY_INDEX_URL.format(year=year, qtr=qtr, ymd=ymd)).decode("latin-1")

    def filing_text_head(self, cik: int, accession_no: str, nbytes: int = 4000) -> str:
        """Start of the full submission text file, which holds the SGML header."""
        self.registry.require("edgar_index", "PARTIAL")
        return self.get(FILING_TXT_URL.format(cik=cik, accession=accession_no))[:nbytes].decode("latin-1")
