"""Odd-lot tender detector.

Why this is a small-account edge: in an issuer tender offer that is oversubscribed, SEC Rule 13e-4
lets the issuer accept in full every holder of fewer than 100 shares before prorating everyone
else. The issuer must say so in the offer document. A fund cannot use it: the benefit is capped at
99 shares per beneficial owner. So the spread between market price and tender price can stay open
for anyone small enough to qualify.

This module reads tender-offer filings (SC TO-I, SC TO-T, SC 13E3 and their 2025+ names) and flags
those whose text grants odd-lot priority. It extracts context, not a trade: price range, expiry
and conditions must be read by a person before anything is done.
"""
from __future__ import annotations

import re

TENDER_FORMS = {"SC TO-I", "SC TO-T", "SC 13E3", "SCHEDULE TO-I", "SCHEDULE TO-T", "SCHEDULE 13E-3",
                "SC TO-I/A", "SC TO-T/A", "SCHEDULE TO-I/A", "SCHEDULE TO-T/A"}

_ODD = re.compile(r"odd[\s\-]?lots?", re.I)
_PRIORITY = re.compile(r"(fewer|less)\s+than\s+100\s+shares|odd[\s\-]?lot\s+(holders?|priority|tenders?)", re.I)
_PRICE = re.compile(r"\$\s?\d{1,4}(?:\.\d{1,4})?\s*(?:per\s+share|net\s+to\s+the\s+seller)", re.I)
_RANGE = re.compile(r"not\s+(?:greater|more)\s+than\s+\$\s?\d+(?:\.\d+)?\s+nor\s+less\s+than\s+\$\s?\d+(?:\.\d+)?"
                    r"|not\s+less\s+than\s+\$\s?\d+(?:\.\d+)?\s+(?:and|nor)\s+not\s+(?:greater|more)\s+than\s+\$\s?\d+(?:\.\d+)?", re.I)


def _clean(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"&nbsp;|&#160;", " ", text)
    return re.sub(r"\s+", " ", text)


def analyse(raw: str) -> dict:
    """Return odd-lot flags and short context snippets from one filing's text."""
    t = _clean(raw)
    odd_hits = [m.start() for m in _ODD.finditer(t)]
    has_priority = bool(_PRIORITY.search(t)) and bool(odd_hits)
    snippet = ""
    for pos in odd_hits:
        window = t[max(0, pos - 220): pos + 260]
        if _PRIORITY.search(window) or "prorat" in window.lower():
            snippet = window.strip()
            break
    prices = sorted({m.group(0).strip() for m in _PRICE.finditer(t)})[:5]
    ranges = sorted({m.group(0).strip() for m in _RANGE.finditer(t)})[:3]
    return {"odd_lot_mentions": len(odd_hits), "odd_lot_priority": has_priority,
            "snippet": snippet[:480], "prices": prices, "price_ranges": ranges}
