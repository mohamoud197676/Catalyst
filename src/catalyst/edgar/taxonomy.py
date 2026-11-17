"""Map SEC form types and 8-K item codes to event classes.

side_hint is a research prior used to route a filing to the long or short watchlist.
It is a hypothesis to test in Phase 1, never a trade signal on its own.
"""
from __future__ import annotations

# 8-K items as defined on SEC Form 8-K.
ITEMS_8K: dict[str, tuple[str, str, str]] = {
    # code: (label, event_class, side_hint)
    "1.01": ("Entry into a material definitive agreement", "MATERIAL_AGREEMENT", "NEUTRAL"),
    "1.02": ("Termination of a material definitive agreement", "AGREEMENT_TERMINATED", "SHORT_WATCH"),
    "1.03": ("Bankruptcy or receivership", "BANKRUPTCY", "SHORT_WATCH"),
    "1.05": ("Material cybersecurity incident", "CYBER_INCIDENT", "SHORT_WATCH"),
    "2.01": ("Completion of acquisition or disposition of assets", "ACQUISITION_COMPLETE", "NEUTRAL"),
    "2.02": ("Results of operations and financial condition", "EARNINGS", "NEUTRAL"),
    "2.03": ("Creation of a direct financial obligation", "NEW_DEBT", "NEUTRAL"),
    "2.04": ("Triggering event accelerating an obligation", "DEBT_ACCELERATION", "SHORT_WATCH"),
    "2.05": ("Costs associated with exit or disposal activities", "RESTRUCTURING", "SHORT_WATCH"),
    "2.06": ("Material impairments", "IMPAIRMENT", "SHORT_WATCH"),
    "3.01": ("Notice of delisting or failure to satisfy a listing rule", "DELISTING_NOTICE", "SHORT_WATCH"),
    "3.02": ("Unregistered sales of equity securities", "UNREGISTERED_EQUITY_SALE", "SHORT_WATCH"),
    "3.03": ("Material modification to rights of security holders", "RIGHTS_MODIFIED", "NEUTRAL"),
    "4.01": ("Change in certifying accountant", "AUDITOR_CHANGE", "SHORT_WATCH"),
    "4.02": ("Non-reliance on previously issued financial statements", "RESTATEMENT", "SHORT_WATCH"),
    "5.01": ("Change in control", "CHANGE_OF_CONTROL", "NEUTRAL"),
    "5.02": ("Departure or appointment of directors or officers", "MANAGEMENT_CHANGE", "NEUTRAL"),
    "5.03": ("Amendment to articles or bylaws", "CHARTER_AMENDMENT", "NEUTRAL"),
    "5.07": ("Submission of matters to a shareholder vote", "SHAREHOLDER_VOTE", "NEUTRAL"),
    "7.01": ("Regulation FD disclosure", "REG_FD", "NEUTRAL"),
    "8.01": ("Other events", "OTHER_EVENT", "NEUTRAL"),
    "9.01": ("Financial statements and exhibits", "EXHIBITS", "NEUTRAL"),
}

# Reverse splits surface through 5.03 (charter amendment) and 3.03; both need text
# extraction in Phase 1 to confirm. They are flagged here so nothing is missed.
REVERSE_SPLIT_CANDIDATE_ITEMS = {"5.03", "3.03"}

FORMS: dict[str, tuple[str, str]] = {
    # base form: (event_class, side_hint)
    "8-K": ("CURRENT_REPORT", "NEUTRAL"),
    "6-K": ("FOREIGN_CURRENT_REPORT", "NEUTRAL"),
    "10-K": ("ANNUAL_REPORT", "NEUTRAL"),
    "10-Q": ("QUARTERLY_REPORT", "NEUTRAL"),
    "20-F": ("ANNUAL_REPORT", "NEUTRAL"),
    "NT 10-K": ("LATE_FILING", "SHORT_WATCH"),
    "NT 10-Q": ("LATE_FILING", "SHORT_WATCH"),
    "S-1": ("REGISTRATION", "SHORT_WATCH"),
    "S-3": ("SHELF_REGISTRATION", "SHORT_WATCH"),
    "S-3ASR": ("SHELF_REGISTRATION", "NEUTRAL"),
    "F-1": ("REGISTRATION", "SHORT_WATCH"),
    "F-3": ("SHELF_REGISTRATION", "SHORT_WATCH"),
    "EFFECT": ("REGISTRATION_EFFECTIVE", "SHORT_WATCH"),
    "424B1": ("OFFERING_PROSPECTUS", "SHORT_WATCH"),
    "424B2": ("OFFERING_PROSPECTUS", "NEUTRAL"),
    "424B3": ("OFFERING_PROSPECTUS", "SHORT_WATCH"),
    "424B4": ("OFFERING_PROSPECTUS", "SHORT_WATCH"),
    "424B5": ("OFFERING_PROSPECTUS", "SHORT_WATCH"),
    "424B7": ("OFFERING_PROSPECTUS", "SHORT_WATCH"),
    "RW": ("REGISTRATION_WITHDRAWN", "NEUTRAL"),
    "3": ("INSIDER_INITIAL", "NEUTRAL"),
    "4": ("INSIDER_TRANSACTION", "NEUTRAL"),
    "5": ("INSIDER_ANNUAL", "NEUTRAL"),
    "144": ("INSIDER_SALE_NOTICE", "SHORT_WATCH"),
    "SC 13D": ("ACTIVIST_STAKE", "LONG_WATCH"),
    "SCHEDULE 13D": ("ACTIVIST_STAKE", "LONG_WATCH"),
    "SC 13G": ("PASSIVE_STAKE", "NEUTRAL"),
    "SCHEDULE 13G": ("PASSIVE_STAKE", "NEUTRAL"),
    "SC TO-T": ("TENDER_OFFER", "LONG_WATCH"),
    "DEFM14A": ("MERGER_PROXY", "NEUTRAL"),
    "25": ("DELISTING", "SHORT_WATCH"),
    "25-NSE": ("DELISTING", "SHORT_WATCH"),
    "15-12B": ("DEREGISTRATION", "SHORT_WATCH"),
    "15-12G": ("DEREGISTRATION", "SHORT_WATCH"),
    "15-15D": ("DEREGISTRATION", "SHORT_WATCH"),
}

# Forms whose filing date stays the acceptance day up to 22:00 ET (ownership reports).
OWNERSHIP_FORMS = {"3", "4", "5"}

# Filings that mark the end of a listing. Used to build the survivorship-free universe.
LISTING_EVENT_BY_CLASS = {
    "DELISTING": "DELIST_NOTICE",          # Form 25: removal takes effect later, date needs confirming
    "DEREGISTRATION": "DEREGISTERED",
    "BANKRUPTCY": "BANKRUPTCY_FILED",
    "DELISTING_NOTICE": "DELIST_NOTICE",
}

_PRIORITY = ["BANKRUPTCY", "RESTATEMENT", "DELISTING_NOTICE", "DEBT_ACCELERATION", "AUDITOR_CHANGE",
             "UNREGISTERED_EQUITY_SALE", "IMPAIRMENT", "EARNINGS", "CHANGE_OF_CONTROL",
             "ACQUISITION_COMPLETE", "MATERIAL_AGREEMENT", "AGREEMENT_TERMINATED"]


def split_form(form_type: str) -> tuple[str, bool]:
    """'S-3/A' -> ('S-3', True)."""
    f = form_type.strip().upper()
    if f.endswith("/A"):
        return f[:-2], True
    return f, False


def classify(form_type: str, items: list[str] | None = None) -> tuple[str, str]:
    """Return (event_class, side_hint) for a filing."""
    base, _ = split_form(form_type)
    if base == "8-K" and items:
        classes = {ITEMS_8K[i][1]: ITEMS_8K[i][2] for i in items if i in ITEMS_8K}
        for cls in _PRIORITY:
            if cls in classes:
                return cls, classes[cls]
    return FORMS.get(base, ("OTHER", "NEUTRAL"))


def parse_items(raw: str | None) -> list[str]:
    if not raw:
        return []
    return [p.strip() for p in raw.split(",") if p.strip()]
