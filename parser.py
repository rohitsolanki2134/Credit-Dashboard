"""
Document parsers: PDF, CSV, Excel → canonical financial dict.

Tuned for Indian annual report PDFs including:
  - PPG Asian Paints  (Rs. in million, Western commas, Note-ref columns, OCR dots)
  - Prestige Estates  (₹ in Mn, integer values, Standalone+Consolidated columns)
  - Generic Indian companies (Rs. in Crores / Lakhs / Millions)

Extraction order (priority high → low):
  1. Detect financial-statement pages (Balance Sheet / P&L / Cash Flow headers)
  2. pdfplumber table scan on those pages
  3. Word-window label scan on those pages (handles single-space columns)
  4. Fall back to full document if < 4 critical fields found
  5. Proximity scan (label line N, value line N+1)
"""

from __future__ import annotations
import io
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from utils import extract_number, get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------

_SCALE_PATTERNS: List[Tuple[re.Pattern, float]] = [
    (re.compile(r"\(?\s*(?:rs\.?|₹|inr)\s+in\s+crores?\b",         re.I), 10_000_000),
    (re.compile(r"\bin\s+(?:rs\.?\s+)?crores?\b",                   re.I), 10_000_000),
    (re.compile(r"(?:amounts?|figures?)\s+in\s+crores?",            re.I), 10_000_000),
    (re.compile(r"\(?\s*(?:rs\.?|₹|inr)\s+in\s+lakhs?\b",          re.I),    100_000),
    (re.compile(r"\bin\s+(?:rs\.?\s+)?lakhs?\b",                    re.I),    100_000),
    (re.compile(r"(?:amounts?|figures?)\s+in\s+lakhs?",             re.I),    100_000),
    (re.compile(r"\(?\s*(?:rs\.?|₹|inr)\s+in\s+(?:millions?|mn)\b", re.I), 1_000_000),
    (re.compile(r"\bin\s+(?:rs\.?\s+)?(?:millions?|mn)\b",          re.I), 1_000_000),
    (re.compile(r"(?:rs\.?|₹)\s*mn\b",                              re.I), 1_000_000),
    (re.compile(r"(?:amounts?|figures?)\s+in\s+(?:millions?|mn)\b", re.I), 1_000_000),
    (re.compile(r"\bin\s+(?:rs\.?\s+)?thousands?\b",                re.I),      1_000),
]


def _detect_scale(text: str) -> float:
    for pat, factor in _SCALE_PATTERNS:
        if pat.search(text):
            log.info("Scale detected: x%s", f"{factor:,}")
            return float(factor)
    return 1.0


# ---------------------------------------------------------------------------
# OCR / formatting cleanup
# ---------------------------------------------------------------------------

_TRAILING_JUNK = re.compile(r"[~#@|\\!]+$")
_NOTE_INLINE   = re.compile(
    r"\b(?:note\s*[\d.]+[\w()]*|[\d]+\([a-z]\)|[\d]+\.[a-z]\b|refer\s+note)\b",
    re.I,
)


def _clean_num_str(s: str) -> str:
    """
    Fix OCR artifacts in a numeric string:
      "17.139.74"  → "17139.74"   (dots as thousands separators)
      "16,154,89"  → "16154.89"   (2-digit trailing comma group = decimal)
      "2.942.21"   → "2942.21"
    """
    s = _TRAILING_JUNK.sub("", s).strip()

    # Multiple dots: all but last are thousands separators
    if s.count(".") > 1:
        parts = s.split(".")
        s = "".join(parts[:-1]) + "." + parts[-1]

    # Trailing 2-digit comma group (continental decimal)
    m = re.match(r"^(-?\(?\d[\d,]*),(\d{2})\)?$", s)
    if m:
        s = m.group(1).replace(",", "") + "." + m.group(2)

    return s


def _is_note_ref(cell: str) -> bool:
    """Return True if a table cell is purely a note/schedule reference."""
    cell = cell.strip()
    return bool(re.match(
        r"^(?:note\s*[\d.]+[\w()]*|[\d]+\([a-z()]+\)|"
        r"[\d]+\.[a-z\d]\w*|schedule\s*\w+|refer\s+note)$",
        cell, re.I,
    ))


# ---------------------------------------------------------------------------
# Field synonyms
# ---------------------------------------------------------------------------

FIELD_SYNONYMS: Dict[str, List[str]] = {
    "revenue": [
        "revenue from operations",
        "net revenue from operations",
        "gross revenue from operations",
        "income from operations",
        "total income from operations",
        "total revenue from operations",
        "net sales and other operating revenues",
        "revenue from contracts with customers",
        "total revenue", "net revenue", "net sales", "total sales",
        "revenues", "sales", "turnover", "total turnover",
        "gross revenue", "total income",
        "total revenue from operations",
    ],
    "gross_profit": [
        "gross profit", "gross income", "gross margin value",
    ],
    "ebitda": [
        "ebidta", "ebitda", "pbdit", "pbitda",
        "profit before depreciation interest and tax",
        "profit before interest depreciation and tax",
        "earnings before depreciation interest and tax",
        "operating profit before depreciation",
        "earnings before interest taxes depreciation amortization",
        "earnings before interest tax depreciation and amortisation",
        "profit before depreciation and amortisation finance costs and tax",
    ],
    "net_profit": [
        "profit for the year", "profit for the period",
        "profit after tax", "pat",
        "profit after taxation",
        "net profit after tax", "net profit for the year",
        "profit attributable to owners",
        "profit attributable to equity holders",
        "profit attributable to equity shareholders",
        "net income", "net profit", "net earnings",
        "income after tax", "net loss",
    ],
    "total_debt": [
        # Only explicit borrowing lines — NOT "total liabilities"
        "total borrowings", "borrowings",
        "long term borrowings", "short term borrowings",
        "long-term borrowings", "short-term borrowings",
        "secured loans", "unsecured loans",
        "total debt", "long-term debt", "short-term debt",
        "notes payable", "bank debt",
        "total interest bearing liabilities",
        "total loans",
    ],
    "equity": [
        "total equity",
        "shareholders funds", "shareholders equity",
        "total shareholders funds",
        "networth", "net worth", "total net worth",
        "equity share capital and reserves",
        "stockholders equity", "total stockholders equity",
        "book value",
    ],
    "current_assets": [
        "total current assets",
        "subtotal current assets",
        # bare "current assets" only if NOT preceded by "non"
        "current assets",
    ],
    "current_liabilities": [
        "total current liabilities",
        "subtotal current liabilities",
        "current liabilities",
    ],
    "cash_and_equivalents": [
        "cash and cash equivalents",
        "cash and bank balances",
        "cash and bank",
        "balances with banks",
        "cash on hand",
        "cash in hand and bank balances",
        "cash and cash equivalents at the end of the year",
        "cash equivalents",
        "cash and short-term investments",
    ],
    "operating_cash_flow": [
        "net cash inflow from operating activities",
        "net cash generated from operating activities",
        "net cash from operating activities",
        "net cash provided by operating activities",
        "net cash inflow outflow from operating activities",
        "cash flow from operations",
        "operating cash flow",
        "cash from operations",
        # "cash generated from operations" intentionally excluded:
        # it is the pre-tax intermediate value, not final net OCF
    ],
    # Internal field used only for current_assets derivation
    "total_noncurrent_assets": [
        "total non-current assets",
        "total noncurrent assets",
        "total non current assets",
    ],
    "interest_expense": [
        "finance costs", "finance cost",
        "interest and finance charges",
        "interest expenses", "borrowing costs",
        "interest expense", "interest cost", "finance charges",
    ],
    "total_assets": [
        "total assets",
        "total equity and liabilities",
        "tomi equity and liabilities",    # common OCR corruption
    ],
    "accounts_receivable": [
        "trade receivables", "accounts receivable",
        "sundry debtors", "debtors", "receivables",
        "trade and other receivables",
    ],
    "accounts_payable": [
        "trade payables", "sundry creditors", "accounts payable",
        "creditors", "trade and other payables",
        "total trade payables", "trade creditors",
    ],
    "inventory": [
        "inventories", "inventory", "stock", "stocks",
    ],
}

# Build reverse map — longer synonyms first so they beat shorter ones
_SYNONYM_MAP: Dict[str, str] = {}
for _field, _syns in FIELD_SYNONYMS.items():
    for _s in _syns:
        _SYNONYM_MAP[_s.lower()] = _field

# Fields that must NOT match if label contains "non-current"
_NO_NONCURRENT = {"current_assets", "current_liabilities"}

# Labels that are balance-sheet sub-components and must NOT match aggregate fields
_LABEL_BLOCKLIST = {
    "equity share capital",
    "paid up equity share capital",
    "other equity",
    "other current assets",
    "other current liabilities",
    "other non current assets",
    "other non current liabilities",
    "other financial assets",
    "other financial liabilities",
    "capital work in progress",
    "capital work-in-progress",
}


_STRIP_PREFIX_TOKENS = {
    "i", "ii", "iii", "iv", "v", "vi", "vii", "viii", "ix", "x",
    "xi", "xii", "xiii", "xiv", "xv",
    "a", "b", "c", "d", "e", "f", "g", "h",
    "1", "2", "3", "4", "5", "6", "7", "8", "9",
}


def _normalise_label(label: str) -> str:
    label = re.sub(r"[*†‡#•]", "", label)
    norm = re.sub(r"[^a-z0-9 ]", " ", label.lower()).strip()
    # Strip leading sub-item markers: "(ii)", "a)", "1." etc.
    parts = norm.split(None, 1)
    if len(parts) == 2 and parts[0] in _STRIP_PREFIX_TOKENS:
        norm = parts[1]
    return norm


def _match_field(label: str) -> Optional[str]:
    norm = _normalise_label(label)
    if not norm or len(norm) < 3:
        return None

    # Hard blocklist — sub-components that must not match aggregate fields
    if norm in _LABEL_BLOCKLIST:
        return None

    # Exact match
    if norm in _SYNONYM_MAP:
        canon = _SYNONYM_MAP[norm]
        if canon in _NO_NONCURRENT and re.search(r"\bnon.?current\b", norm):
            return None
        return canon

    # startswith match: normalised label must start with a synonym.
    # Single-word synonyms (e.g. "revenues", "sales") must be exact matches
    # only — the exact-match branch above already handles them.  Allowing
    # startswith for single-word synonyms turns "revenues of `" into a revenue
    # hit because it starts with "revenues".
    for syn in sorted(_SYNONYM_MAP, key=len, reverse=True):
        if norm.startswith(syn):
            if len(syn.split()) == 1 and norm != syn:
                continue  # single-word synonym must be exact
            canon = _SYNONYM_MAP[syn]
            if canon in _NO_NONCURRENT and re.search(r"\bnon.?current\b", norm):
                continue
            return canon
    return None


# ---------------------------------------------------------------------------
# Number extraction from multi-value cells / strings
# ---------------------------------------------------------------------------

# Matches a single number token (handles Indian/Western commas + optional suffix)
_NUM_TOKEN_RE = re.compile(
    r"[-+]?\(?\s*"
    r"((?:\d[\d,.]*\d|\d))"   # the digits (with internal commas/dots)
    r"\s*\)?"
    r"(?:\s*(?:crores?|cr|lakhs?|lacs?|mn|millions?))?",
    re.I,
)


def _extract_all_nums(text: str) -> List[float]:
    """
    Extract every valid number from a string, cleaning OCR artifacts.
    Handles "17.139.74" as a single number (17139.74).
    """
    # First: replace multi-dot sequences (OCR artifact) with cleaned form
    # "17.139.74" → "17139.74", "2.942.21" → "2942.21"
    text = re.sub(
        r"(\d+)\.(\d{3})\.(\d{2})\b",
        lambda m: m.group(1) + m.group(2) + "." + m.group(3),
        text,
    )
    text = re.sub(
        r"(\d+)\.(\d{3})\.(\d{3})\b",
        lambda m: m.group(1) + m.group(2) + m.group(3),
        text,
    )

    nums: List[float] = []
    for m in _NUM_TOKEN_RE.finditer(text):
        raw = _clean_num_str(m.group(0).strip())
        # Parse the cleaned string directly — extract_number's _NUMBER_RE
        # has a 1-3 digit greedy group that truncates numbers like "17139.74"
        # to 171.0.  After _clean_num_str the string is already normalised.
        # Strip trailing unit suffixes — the PDF scale (Cr/Mn/etc.) is applied
        # by the caller; letting extract_number multiply them in again would
        # double-count (e.g. "46,469 million" × scale × 1e6).
        is_neg = raw.startswith("(") and ")" in raw
        clean_raw = re.sub(
            r"\s*(?:crores?|cr|lakhs?|lacs?|mn|millions?|billions?|thousands?)\s*$",
            "", raw, flags=re.I,
        )
        try:
            n = float(re.sub(r"[(),\s]", "", clean_raw))
            if is_neg:
                n = -n
            nums.append(n)
        except (ValueError, TypeError):
            n = extract_number(raw)
            if n is not None:
                nums.append(n)
    return nums


def _best_value(text: str) -> Optional[float]:
    """
    Return the most likely current-year value from a multi-value string.

    Indian annual reports follow: [Label] [Note#] [FY25_value] [FY24_value]
    Strategy:
      1. Strip inline note references (e.g. "note 15")
      2. Filter bare note-column integers that are < 1% of the max value
      3. Return the FIRST remaining number (= current/most-recent year)
    """
    clean = _NOTE_INLINE.sub(" ", text)
    nums = _extract_all_nums(clean)
    if not nums:
        return None
    if len(nums) == 1:
        return nums[0]

    # Filter note-column integers: a bare integer is a note ref if it is
    # < 1% of the maximum absolute value present (e.g. "38" vs 73,494).
    max_abs = max(abs(n) for n in nums)
    if max_abs > 0:
        threshold = max_abs * 0.01
        filtered = [n for n in nums if abs(n) >= threshold or n != int(n)]
        if filtered:
            nums = filtered

    return nums[0]  # first value = current fiscal year


# ---------------------------------------------------------------------------
# Financial-page detection
# ---------------------------------------------------------------------------

_FIN_PAGE_MARKERS = [
    "balance sheet as at",
    "consolidated balance sheet",
    "standalone balance sheet",
    "statement of profit and loss",
    "statement of profit & loss",
    "statement of income",
    "cash flow statement",
    "statement of cash flows",
    "consolidated statement of profit",
    "standalone statement of profit",
    "profit and loss account",
]


def _financial_page_indices(pdf) -> List[int]:
    """
    Return indices of financial statement pages, consolidated first.

    When a PDF contains both standalone and consolidated statements (common in
    Indian annual reports), consolidated pages are returned before standalone
    so the parser locks in consolidated figures first.
    """
    consolidated: List[int] = []
    standalone: List[int] = []
    other_hits: List[int] = []

    for i, page in enumerate(pdf.pages):
        text = (page.extract_text() or "").lower()
        if not any(m in text for m in _FIN_PAGE_MARKERS):
            continue
        # Classify by presence of "consolidated" / "standalone" in first 600 chars
        header = text[:600]
        if "consolidated" in header:
            consolidated.append(i)
        elif "standalone" in header:
            standalone.append(i)
        else:
            other_hits.append(i)

    # Priority: consolidated > standalone > other
    priority = consolidated or standalone or other_hits
    if not priority:
        return []

    extended: set = set()
    for idx in priority:
        extended.update(range(idx, min(idx + 4, len(pdf.pages))))

    log.info("Fin-page priority: %d consolidated, %d standalone, %d other → %d total",
             len(consolidated), len(standalone), len(other_hits), len(extended))
    return sorted(extended)


# ---------------------------------------------------------------------------
# PDF parser — pdfplumber (primary)
# ---------------------------------------------------------------------------

def parse_pdf(raw_bytes: bytes, company_name: str = "") -> Tuple[dict, float, list]:
    try:
        import pdfplumber
    except ImportError:
        return _parse_pdf_pymupdf(raw_bytes, company_name)

    data: Dict[str, Any] = {"company_name": company_name}
    found_fields: set = set()

    with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
        total = len(pdf.pages)
        log.info("PDF opened: %d pages", total)

        # --- Collect first-50-page text for scale detection + company name ---
        head_lines: List[str] = []
        for page in pdf.pages[:50]:
            head_lines.extend((page.extract_text() or "").splitlines())

        # Scale: scan header area and per-page snippets
        scale = _detect_scale("\n".join(head_lines))
        if scale == 1.0:
            all_head = "\n".join(head_lines)
            scale = _detect_scale(all_head)
        # Deeper scan if still not found
        if scale == 1.0:
            for page in pdf.pages[50:]:
                snippet = (page.extract_text() or "")[:1000]
                s = _detect_scale(snippet)
                if s != 1.0:
                    scale = s
                    break

        if not company_name:
            data["company_name"] = _extract_company_name(head_lines, pdf)

        # --- Find formal financial statement pages ---
        fin_pages = _financial_page_indices(pdf)
        log.info("Financial statement pages: %s", fin_pages[:10])

        if not fin_pages:
            fin_pages = list(range(total))  # fallback: scan all

        # --- Pass 1a: text scan on financial pages (runs BEFORE table scan) ---
        # Text extraction is more reliable for OCR-heavy PDFs; locking in
        # correct values first prevents wrong table rows from overwriting them.
        fin_lines: List[str] = []
        for i in fin_pages:
            page = pdf.pages[i]
            text = page.extract_text(x_tolerance=3, y_tolerance=3) or ""
            fin_lines.extend(text.splitlines())

        _word_window_scan(fin_lines, data, found_fields, scale)
        _proximity_scan(fin_lines, data, found_fields, scale)

        # --- Pass 1b: table scan on financial pages (fills in what text missed) ---
        for i in fin_pages:
            page = pdf.pages[i]
            for strategy in [
                {},
                {"vertical_strategy": "text", "horizontal_strategy": "text"},
            ]:
                try:
                    tables = page.extract_tables(strategy) or []
                except Exception:
                    tables = []
                for table in tables:
                    _scan_table(table, data, found_fields, scale)

        # --- Pass 2: full document fallback if still missing critical fields ---
        critical = {"revenue", "net_profit", "equity", "current_assets", "current_liabilities"}
        # current_assets may be derived in post-process; check without it first
        check_critical = critical - {"current_assets"}
        if not check_critical.issubset(found_fields):
            log.info("Pass 2: full-document scan (missing: %s)",
                     critical - found_fields)
            all_lines: List[str] = []
            for page in pdf.pages:
                all_lines.extend((page.extract_text() or "").splitlines())
            _word_window_scan(all_lines, data, found_fields, scale)
            _proximity_scan(all_lines, data, found_fields, scale)

    # Post-process: derive missing balance sheet fields
    # 1. current_assets = total_assets − non-current_assets
    if "current_assets" not in found_fields:
        ta = data.get("total_assets") or 0
        nca = data.get("total_noncurrent_assets") or 0
        if ta > 0 and nca > 0 and ta > nca:
            data["current_assets"] = ta - nca
            found_fields.add("current_assets")
            log.info("Derived current_assets = %.4g - %.4g = %.4g", ta, nca, ta - nca)

    # 2. total_assets from equity + liabilities if still missing
    if "total_assets" not in found_fields:
        eq = data.get("equity") or 0
        td = data.get("total_debt") or 0
        cl = data.get("current_liabilities") or 0
        if eq > 0:
            data["total_assets"] = eq + td + cl
            found_fields.add("total_assets")

    # Remove internal-only field from output
    data.pop("total_noncurrent_assets", None)

    confidence = _confidence_score(data)
    warnings   = _build_warnings(data, confidence)
    log.info("PDF parse done | scale=x%s | fields=%d | confidence=%.0f%%",
             f"{scale:,.0f}", len(found_fields), confidence * 100)
    return data, confidence, warnings


# ---------------------------------------------------------------------------
# PyMuPDF fallback
# ---------------------------------------------------------------------------

def _parse_pdf_pymupdf(raw_bytes: bytes, company_name: str) -> Tuple[dict, float, list]:
    try:
        import fitz
    except ImportError:
        raise ImportError("Install pdfplumber or PyMuPDF: pip install pdfplumber pymupdf")

    data: Dict[str, Any] = {"company_name": company_name}
    found_fields: set = set()
    all_lines: List[str] = []

    doc = fitz.open(stream=raw_bytes, filetype="pdf")
    for page in doc:
        for b in page.get_text("blocks"):
            all_lines.extend(str(b[4]).splitlines())

    scale = _detect_scale("\n".join(all_lines[:200]))
    if scale == 1.0:
        scale = _detect_scale("\n".join(all_lines))

    if not company_name:
        data["company_name"] = _extract_company_name(all_lines)

    _word_window_scan(all_lines, data, found_fields, scale)
    _proximity_scan(all_lines, data, found_fields, scale)

    confidence = _confidence_score(data)
    log.info("PyMuPDF parse | scale=x%s | fields=%d | confidence=%.0f%%",
             f"{scale:,.0f}", len(found_fields), confidence * 100)
    return data, confidence, _build_warnings(data, confidence)


# ---------------------------------------------------------------------------
# Scan strategies
# ---------------------------------------------------------------------------

def _scan_table(table: list, data: dict, found_fields: set, scale: float) -> None:
    """
    Scan a pdfplumber table row by row.

    Tries EVERY cell as a potential label, not just row[0].  This handles
    two-column layouts (e.g. Prestige's Balance Sheet + P&L on the same page)
    where the P&L items appear in the middle columns of the merged table.
    """
    for row in table:
        if not row or len(row) < 2:
            continue

        matched_this_row = False
        for label_idx, label_cell in enumerate(row[:-1]):
            if not label_cell:
                continue
            label_text = str(label_cell).strip()
            label_lines = [l for l in label_text.splitlines() if l.strip()]
            if not label_lines:
                continue
            label = label_lines[0]

            canonical = _match_field(label)
            if not canonical or canonical in found_fields:
                continue

            # Values are cells to the RIGHT of this label cell
            value_cells = [str(c) for c in row[label_idx + 1:] if c is not None]
            value_cells = [c for c in value_cells if not _is_note_ref(c.strip())]
            combined = " ".join(value_cells)
            val = _best_value(combined)
            if val is not None:
                data[canonical] = val * scale
                found_fields.add(canonical)
                log.debug("TABLE      | %-28s = %s (raw %.4g) [label='%s']",
                          canonical, val * scale, val, label[:40])
                matched_this_row = True
                break  # one canonical per row is enough


def _word_window_scan(lines: List[str], data: dict, found_fields: set, scale: float) -> None:
    """
    Sliding word-window: try each prefix of 1-8 words as a label.
    This handles both "Revenue from operations   21,423.23" (2+ spaces)
    and "Total equity 12,159.98 11,662.69" (single spaces).
    """
    for line in lines:
        line = line.strip()
        if not line or len(line) < 4:
            continue

        words = line.split()
        if len(words) < 2:
            continue

        for n in range(min(len(words) - 1, 12), 0, -1):
            # Skip if the last word of the prefix contains a digit —
            # that means we've already consumed a numeric value into the label.
            if re.search(r'\d', words[n - 1]):
                continue
            label = " ".join(words[:n])
            canonical = _match_field(label)
            if not canonical or canonical in found_fields:
                continue

            value_part = " ".join(words[n:])
            # Strip inline note references
            value_part = _NOTE_INLINE.sub(" ", value_part)
            # Reject if value part doesn't start with a numeric-ish token.
            # A label like "revenues" could match the start of a narrative
            # sentence "revenues of ₹ 46,469 million..." — the value part
            # "of ₹ 46,469..." starts with a word, not a number.
            if not re.match(r'^[-+\d₹\$\(]', value_part.strip()):
                continue
            val = _best_value(value_part)
            if val is not None:
                data[canonical] = val * scale
                found_fields.add(canonical)
                log.debug("WORD-WIN   | %-28s = %s (label='%s')", canonical, val * scale, label)
                break   # stop trying shorter labels once a match is found


def _proximity_scan(lines: List[str], data: dict, found_fields: set, scale: float) -> None:
    """Label on line N, value on line N+1 (narrow PDF layouts)."""
    for i, line in enumerate(lines[:-1]):
        label = line.strip()
        canonical = _match_field(label)
        if not canonical or canonical in found_fields:
            continue
        nxt = _clean_num_str(lines[i + 1].strip())
        if re.match(r"^[-+₹(]?\s*[\d,.]+(?:\s*(?:cr|crore|lakh|lac|mn|million))?$", nxt, re.I):
            val = extract_number(nxt)
            if val is not None:
                data[canonical] = val * scale
                found_fields.add(canonical)
                log.debug("PROXIMITY  | %-28s = %s", canonical, val * scale)


# ---------------------------------------------------------------------------
# Company name extraction
# ---------------------------------------------------------------------------

_SKIP_HDR = re.compile(
    r"^\s*(?:"
    r"balance\s+sheet|profit\s+and\s+loss|cash\s+flow|notes?\s+to|"
    r"statement\s+of|consolidated|standalone|auditor|board\s+of|"
    r"annual\s+report|financial\s+year|rs\.|inr|₹|\d|page\s+\d|"
    r"statutory|corporate|overview|national\s+stock|bombay\s+stock|bse|nse|"
    r"stock\s+exchange|cin:|registered"
    r")",
    re.I,
)


def _extract_company_name(lines: List[str], pdf=None) -> str:
    # Priority: lines ending in Ltd/Limited/Pvt/LLP
    for line in lines[:80]:
        line = line.strip()
        if (re.search(r"\b(ltd\.?|limited|pvt\.?|private|llp|inc\.?)\b", line, re.I)
                and not _SKIP_HDR.match(line)
                and 8 < len(line) < 120):
            return line
    # Fallback: first meaningful capitalised line
    for line in lines[:30]:
        line = line.strip()
        if (len(line) > 6
                and not _SKIP_HDR.match(line)
                and line[0].isupper()
                and not line[0].isdigit()
                and sum(c.isalpha() for c in line) > len(line) * 0.5):
            return line
    return ""


# ---------------------------------------------------------------------------
# CSV / Excel parsers
# ---------------------------------------------------------------------------

def parse_csv(raw_bytes: bytes) -> Tuple[dict, float, list]:
    df = pd.read_csv(io.BytesIO(raw_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    return _parse_dataframe(df)


def parse_excel(raw_bytes: bytes) -> Tuple[dict, float, list]:
    df = pd.read_excel(io.BytesIO(raw_bytes))
    df.columns = [str(c).strip() for c in df.columns]
    return _parse_dataframe(df)


def _parse_dataframe(df: pd.DataFrame) -> Tuple[dict, float, list]:
    data: Dict[str, Any] = {}
    found_fields: set = set()
    lower_cols = [c.lower() for c in df.columns]

    if any(k in lower_cols for k in ("metric", "field", "item", "label")):
        label_col = next(c for c in df.columns if c.lower() in ("metric", "field", "item", "label"))
        value_col = next(
            (c for c in df.columns if c.lower() in ("value", "amount", "figure")),
            df.columns[1] if len(df.columns) > 1 else None,
        )
        if value_col is None:
            return data, 0.0, ["Could not identify value column"]
        for _, row in df.iterrows():
            canonical = _match_field(str(row[label_col]))
            if canonical and canonical not in found_fields:
                n = extract_number(_clean_num_str(str(row[value_col])))
                if n is not None:
                    data[canonical] = n
                    found_fields.add(canonical)
    else:
        for col in df.columns:
            canonical = _match_field(col)
            if canonical and canonical not in found_fields:
                vals = df[col].dropna()
                if len(vals) > 0:
                    n = extract_number(_clean_num_str(str(vals.iloc[-1])))
                    if n is not None:
                        data[canonical] = n
                        found_fields.add(canonical)
        for col in df.columns:
            if "company" in col.lower() or "name" in col.lower():
                data["company_name"] = str(df[col].iloc[-1])
                break

    for col in df.columns:
        if any(k in col.lower() for k in ("year", "period", "date", "fy")):
            data["fiscal_year"] = str(df[col].iloc[-1])
            break

    confidence = _confidence_score(data)
    return data, confidence, _build_warnings(data, confidence)


# ---------------------------------------------------------------------------
# Confidence scoring
# ---------------------------------------------------------------------------

_CRITICAL_FIELDS = [
    "revenue", "net_profit", "total_debt", "equity",
    "current_assets", "current_liabilities",
]
_IMPORTANT_FIELDS = [
    "ebitda", "operating_cash_flow", "interest_expense",
    "cash_and_equivalents", "total_assets",
]


def _confidence_score(data: dict) -> float:
    c = sum(1 for f in _CRITICAL_FIELDS  if data.get(f) is not None)
    i = sum(1 for f in _IMPORTANT_FIELDS if data.get(f) is not None)
    return round(c / len(_CRITICAL_FIELDS) * 0.70 + i / len(_IMPORTANT_FIELDS) * 0.30, 3)


def _build_warnings(data: dict, confidence: float) -> list:
    warnings: list = []
    for f in _CRITICAL_FIELDS:
        if data.get(f) is None:
            warnings.append(f"Could not extract '{f}' — please enter manually")
    if confidence < 0.5:
        warnings.append(
            f"Low parse confidence ({confidence:.0%}). "
            "Review extracted values and correct any errors before proceeding."
        )
    return warnings
