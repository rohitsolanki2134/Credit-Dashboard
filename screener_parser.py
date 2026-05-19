"""
Screener.in data fetcher — pulls financial data for NSE/BSE listed companies.

Usage:
    results = search_company("Infosys")
    # → [{"name": "Infosys Ltd", "symbol": "INFY", "url": "..."}]

    financials, warnings = fetch_financials("INFY", consolidated=True)
    # → ({"revenue": 1786500000000, "net_profit": ...}, ["..."])
"""

from __future__ import annotations
import re
from typing import Optional
from utils import get_logger

log = get_logger(__name__)

_BASE    = "https://www.screener.in"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Referer":         "https://www.screener.in/",
}


# ---------------------------------------------------------------------------
# Public: Company search
# ---------------------------------------------------------------------------

def search_company(query: str) -> list[dict]:
    """
    Search Screener.in for companies matching `query`.
    Returns a list of dicts: {name, symbol, url}.
    Correct API endpoint: /api/company/search/?q=<query>
    """
    try:
        import requests
        resp = requests.get(
            f"{_BASE}/api/company/search/",
            params={"q": query.strip()},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
        raw = resp.json()
        items = raw if isinstance(raw, list) else raw.get("results", [])
        results = []
        for item in items:
            url    = item.get("url", "")
            # URL format: /company/SYMBOL/consolidated/  or  /company/SYMBOL/
            parts  = [p for p in url.strip("/").split("/") if p and p != "consolidated"]
            symbol = parts[-1].upper() if parts else item.get("name", "")
            results.append({
                "name":   item.get("name", symbol),
                "symbol": symbol,
                "url":    url,
            })
        return results[:10]
    except Exception as exc:
        log.error("Screener search error: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Public: Fetch financials
# ---------------------------------------------------------------------------

def fetch_financials(
    symbol: str,
    consolidated: bool = True,
    scale: float = 10_000_000,   # Screener shows ₹ Crores by default
) -> tuple[dict, list[str]]:
    """
    Fetch P&L, Balance Sheet, and Cash Flow data from Screener.in for `symbol`.
    Returns (financials_dict, warnings_list).
    All monetary values are converted to raw INR using `scale`.
    """
    try:
        from bs4 import BeautifulSoup
        import requests
    except ImportError:
        return {}, [
            "Missing dependency — run:  pip install beautifulsoup4 requests"
        ]

    suffix   = "consolidated/" if consolidated else ""
    url      = f"{_BASE}/company/{symbol.upper()}/{suffix}"
    warnings: list[str] = []

    try:
        resp = requests.get(url, headers=_HEADERS, timeout=15)
        # If consolidated not found, silently fall back to standalone
        if resp.status_code == 404 or "Page not found" in resp.text[:500]:
            if consolidated:
                url  = f"{_BASE}/company/{symbol.upper()}/"
                resp = requests.get(url, headers=_HEADERS, timeout=15)
                warnings.append(
                    "Consolidated financials not available — using standalone data."
                )
        resp.raise_for_status()
    except Exception as exc:
        return {}, [f"Could not reach Screener.in for '{symbol}': {exc}"]

    if "Page not found" in resp.text[:500]:
        return {}, [
            f"Company '{symbol}' not found on Screener.in. "
            "Check the ticker and try again."
        ]

    soup = BeautifulSoup(resp.text, "html.parser")

    # ── Company name ────────────────────────────────────────────────────────
    company_name = symbol
    for selector in [("h1", {"class": "h2"}), ("h1", {}), ("h2", {})]:
        el = soup.find(selector[0], selector[1])
        if el:
            txt = el.get_text(strip=True)
            if txt and len(txt) > 1:
                company_name = txt
                break

    # ── Generic table parser ────────────────────────────────────────────────
    def _parse_section(section_id: str) -> tuple[dict[str, list], list[str]]:
        """Return ({label: [float|None, ...]}, [year_headers])."""
        section = soup.find("section", {"id": section_id})
        if not section:
            return {}, []
        table = section.find("table")
        if not table:
            return {}, []

        # Year headers from thead
        years: list[str] = []
        thead = table.find("thead")
        if thead:
            years = [
                th.get_text(strip=True)
                for th in thead.find_all("th")
            ][1:]   # skip blank first column

        rows: dict[str, list] = {}
        for tr in (table.find("tbody") or table).find_all("tr"):
            cells = tr.find_all(["td", "th"])
            if len(cells) < 2:
                continue
            # Strip trailing "+" and whitespace from label
            label = cells[0].get_text(strip=True).rstrip("+").strip()
            vals  = [_parse_num(c.get_text(strip=True)) for c in cells[1:]]
            if label:
                rows[label] = vals
        return rows, years

    def _parse_num(text: str) -> Optional[float]:
        """Parse a numeric cell (handles commas, brackets for negatives, % signs)."""
        t = text.strip().replace(",", "").replace("\xa0", "").replace("%", "")
        if not t or t in ("-", "—", ""):
            return None
        if t.startswith("(") and t.endswith(")"):
            t = "-" + t[1:-1]
        try:
            return float(t)
        except ValueError:
            return None

    def _get(rows: dict, *labels) -> tuple[Optional[float], Optional[float]]:
        """
        Find the best matching row and return (current_year_value, prior_year_value).
        Matches are case-insensitive substring checks.
        Returns the two most recent non-None values.
        """
        for lbl in labels:
            lbl_lo = lbl.lower()
            for key, vals in rows.items():
                if lbl_lo in key.lower():
                    nums = [v for v in vals if v is not None]
                    cy   = nums[-1] if nums else None
                    py   = nums[-2] if len(nums) >= 2 else None
                    return cy, py
        return None, None

    # ── Parse the three financial sections ──────────────────────────────────
    pl, pl_years = _parse_section("profit-loss")
    bs, bs_years = _parse_section("balance-sheet")
    cf, cf_years = _parse_section("cash-flow")

    # ── P & L mappings ───────────────────────────────────────────────────────
    # "Sales+" row label becomes "Sales" after stripping
    rev_cy,    rev_py    = _get(pl, "Sales", "Revenue from Operations", "Net Sales", "Total Revenue", "Revenue")
    ebitda_cy, ebitda_py = _get(pl, "Operating Profit", "EBITDA", "PBDIT", "PBIDT")
    np_cy,     np_py     = _get(pl, "Net Profit", "PAT", "Profit after tax", "Profit/Loss After Tax")
    int_cy,    int_py    = _get(pl, "Interest", "Finance Cost", "Finance Costs")

    # If Net Profit is missing, try to derive: PBT × (1 - tax%)
    if np_cy is None:
        pbt_cy, pbt_py = _get(pl, "Profit before tax", "PBT")
        # Tax% row
        tax_cy, tax_py = _get(pl, "Tax %", "Tax Rate")
        if pbt_cy is not None and tax_cy is not None:
            np_cy = pbt_cy * (1 - tax_cy / 100)
        if pbt_py is not None and tax_py is not None:
            np_py = pbt_py * (1 - tax_py / 100)

    # ── Balance Sheet mappings ───────────────────────────────────────────────
    eq_cap_cy,  eq_cap_py  = _get(bs, "Equity Capital", "Share Capital", "Paid up Capital")
    reserves_cy, reserves_py = _get(bs, "Reserves", "Reserves and Surplus", "Other Equity")
    # Total equity = share capital + reserves
    eq_cy = ((eq_cap_cy or 0) + (reserves_cy or 0)) or None
    eq_py = ((eq_cap_py or 0) + (reserves_py or 0)) or None

    debt_cy,  debt_py  = _get(bs, "Borrowings", "Total Debt", "Total Borrowings", "Debt")
    ca_cy,    ca_py    = _get(bs, "Total Current Assets", "Current Assets", "Other Assets")
    cl_cy,    cl_py    = _get(bs, "Total Current Liabilities", "Current Liabilities", "Other Liabilities")
    cash_cy,  cash_py  = _get(bs, "Cash & Bank", "Cash and Cash Equivalents", "Cash Equivalents", "Cash")
    ar_cy,    ar_py    = _get(bs, "Trade Receivables", "Debtors", "Receivables", "Accounts Receivable")
    ap_cy,    ap_py    = _get(bs, "Trade Payables", "Creditors", "Payables", "Accounts Payable")

    # ── Cash Flow mappings ───────────────────────────────────────────────────
    ocf_cy, ocf_py = _get(cf,
        "Cash from Operating Activity",
        "Cash from Operating Activities",
        "Operating Activities",
        "Cash from Operations",
        "CFO",
    )

    # ── Financial year labels ────────────────────────────────────────────────
    fy_current = pl_years[-1] if pl_years else (bs_years[-1] if bs_years else "")
    fy_prior   = (pl_years[-2] if len(pl_years) >= 2
                  else (bs_years[-2] if len(bs_years) >= 2 else ""))

    # ── Convert Crores → raw INR and build result dict ───────────────────────
    def _cr(v: Optional[float]) -> Optional[float]:
        return v * scale if v is not None else None

    fin: dict = {"company_name": company_name}
    if fy_current:
        fin["financial_year"] = fy_current
    if fy_prior:
        fin["prior_financial_year"] = fy_prior

    _field_map = {
        "revenue":                 rev_cy,
        "ebitda":                  ebitda_cy,
        "net_profit":              np_cy,
        "equity":                  eq_cy,
        "total_debt":              debt_cy,
        "current_assets":          ca_cy,
        "current_liabilities":     cl_cy,
        "cash_and_equivalents":    cash_cy,
        "accounts_receivable":     ar_cy,
        "accounts_payable":        ap_cy,
        "operating_cash_flow":     ocf_cy,
        "interest_expense":        int_cy,
        # Prior year
        "revenue_py":              rev_py,
        "ebitda_py":               ebitda_py,
        "net_profit_py":           np_py,
        "equity_py":               eq_py,
        "total_debt_py":           debt_py,
        "current_assets_py":       ca_py,
        "current_liabilities_py":  cl_py,
        "cash_and_equivalents_py": cash_py,
        "accounts_receivable_py":  ar_py,
        "accounts_payable_py":     ap_py,
        "operating_cash_flow_py":  ocf_py,
        "interest_expense_py":     int_py,
    }
    for key, val in _field_map.items():
        converted = _cr(val)
        if converted is not None:
            fin[key] = converted

    # ── Quality warnings ────────────────────────────────────────────────────
    core_fields   = ["revenue", "net_profit", "equity", "total_debt", "operating_cash_flow"]
    missing_core  = [k for k in core_fields if k not in fin]
    if missing_core:
        warnings.append(
            f"⚠️ Could not extract: **{', '.join(missing_core)}**. "
            "Please fill these in the preview before running the evaluation."
        )

    extracted_count = sum(1 for k in _field_map if k in fin)
    py_count        = sum(1 for k in _field_map if k.endswith("_py") and k in fin)

    warnings.append(
        f"✅ **{extracted_count} fields** extracted from Screener.in for "
        f"**{company_name}** (year: {fy_current or 'latest'}, scale: ₹ Crores). "
        "Verify figures match the latest annual report before proceeding."
    )
    if py_count > 0:
        warnings.append(
            f"📅 Prior-year data ({fy_prior}) extracted for {py_count} fields — "
            "trend analysis enabled."
        )
    else:
        warnings.append(
            "⚠️ No prior-year data found. Add prior-year values in the preview to enable trend analysis."
        )

    log.info(
        "Screener fetch: symbol=%s company=%s fy=%s fields=%d py_fields=%d",
        symbol, company_name, fy_current, extracted_count, py_count,
    )
    return fin, warnings
