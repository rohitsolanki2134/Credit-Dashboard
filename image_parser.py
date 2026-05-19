"""
Image-based financial table parser.

Uses Tesseract OCR with bounding-box data to reconstruct a two-column
annual-report table (current year + prior year) without any API key.

Pipeline:
  1. Pre-process image  — greyscale, denoise, threshold for OCR accuracy
  2. OCR with word boxes — pytesseract.image_to_data gives (text, x, y, w, h)
  3. Cluster into rows   — group words whose y-centres are within ~10 px
  4. Identify columns    — words cluster into ≤3 x-bands: label | CY | PY
  5. Parse numbers       — handle commas, spaces, negatives, OCR artifacts
  6. Match field names   — keyword synonyms with normalisation
  7. Detect scale        — Cr / Lakh / Mn → multiply
"""

from __future__ import annotations
import io
import re
from typing import Dict, List, Optional, Tuple

from utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Field synonym map  (lowercase, stripped)
# ---------------------------------------------------------------------------

_FIELD_MAP: Dict[str, List[str]] = {
    "revenue": [
        "annual revenue", "revenue", "total revenue", "net revenue",
        "turnover", "net sales", "sales", "income from operations",
        "total income",
    ],
    "ebitda": ["ebitda", "ebidta"],
    "net_profit": [
        "net profit", "net loss", "net profit / (loss)", "net profit/(loss)",
        "profit after tax", "pat", "profit / (loss)", "profit/(loss)",
        "profit after tax (pat)", "net income",
    ],
    "current_assets": [
        "current assets", "total current assets",
    ],
    "current_liabilities": [
        "current liabilities", "total current liabilities",
        "current liab",          # abbreviated form e.g. "Current Liab."
    ],
    "total_debt": [
        "total debt", "total borrowings", "total debt / borrowings",
        "borrowings", "total debt/borrowings", "debt",
    ],
    "equity": [
        "total net worth", "net worth", "total equity", "shareholders equity",
        "shareholders funds", "shareholder equity", "equity",
        "total shareholders equity", "net worth / equity",
        "net worth equity",
    ],
    "cash_and_equivalents": [
        "cash & bank balance", "cash and bank balance", "cash & bank balances",
        "cash and bank balances", "cash & equivalents",
        "cash and cash equivalents",
        "cash & bank",           # short form used in summary tables
        "cash and bank",
    ],
    "accounts_receivable": [
        "accounts receivable", "accounts receivables",
        "trade receivables", "debtors",
        "receivables",           # short form
        "trade receivable",
    ],
    "accounts_payable": [
        "accounts payable", "accounts payables",
        "trade payables", "creditors",
        "payables",              # short form
        "trade payable",
    ],
    "operating_cash_flow": [
        "operational cash flow", "operating cash flow",
        "cash flow from operations", "net cash from operations",
        "cash from operating activities",
        "ops cash flow",         # abbreviated — matches this dashboard's image format
        "ops. cash flow",
        "operating cf",
        "net operating cash flow",
    ],
    "interest_expense": [
        "interest expense", "finance costs", "finance cost",
        "interest & finance charges", "interest cost",
        "finance charges",
    ],
    "travel_cost": [
        "travel spend", "travel cost", "travel & conveyance",
        "travel and conveyance", "conveyance", "travel expenses",
        "travel",                # very short — only matched if nothing else fits
    ],
}

# Pre-build reverse lookup: synonym → canonical field
_SYN_LOOKUP: Dict[str, str] = {}
for _field, _syns in _FIELD_MAP.items():
    for _s in _syns:
        _SYN_LOOKUP[_s] = _field


# ---------------------------------------------------------------------------
# Number cleaning
# ---------------------------------------------------------------------------

# Common OCR substitutions that corrupt numbers
_OCR_DIGIT_FIX = str.maketrans({
    "O": "0", "o": "0",
    "l": "1", "I": "1",
    "S": "5",
    "Z": "2",
    "B": "8",
})

_NUM_RE = re.compile(r"-?\s*[\d,]+(?:\.\d+)?")


def _parse_number(raw: str) -> Optional[float]:
    """
    Parse a potentially OCR-mangled number string.
    Handles: commas, spaces between digits, OCR letter substitutions,
    leading/trailing junk, and parenthesised negatives like (271.19).
    """
    s = raw.strip()
    negative = s.startswith("-") or (s.startswith("(") and s.endswith(")"))
    # Strip parens
    s = s.strip("()")
    # Apply digit fixes
    s = s.translate(_OCR_DIGIT_FIX)
    # Remove anything that isn't digit, dot, comma, minus
    s = re.sub(r"[^\d.,-]", "", s)
    # Remove commas (thousands separator)
    s = s.replace(",", "")
    # Multiple dots → keep only last (common OCR artifact)
    parts = s.split(".")
    if len(parts) > 2:
        s = parts[0] + "." + parts[-1]
    s = s.strip("-")
    try:
        val = float(s)
        return -val if negative else val
    except ValueError:
        return None


def _extract_numbers_from_line(token: str) -> List[float]:
    """Return all valid numbers found in a text token."""
    results = []
    for m in _NUM_RE.finditer(token):
        v = _parse_number(m.group())
        if v is not None:
            results.append(v)
    return results


# ---------------------------------------------------------------------------
# Label matching
# ---------------------------------------------------------------------------

def _normalise_label(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation noise."""
    t = text.lower()
    t = re.sub(r"[^a-z0-9 &/()\-]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _match_label(raw_label: str) -> Optional[str]:
    """Return canonical field name or None."""
    norm = _normalise_label(raw_label)
    # Exact
    if norm in _SYN_LOOKUP:
        return _SYN_LOOKUP[norm]
    # Contains
    for syn, field in _SYN_LOOKUP.items():
        if syn in norm or norm in syn:
            # Avoid very short matches (e.g. "cash" matching "cash & bank balance")
            if len(syn) >= 5:
                return field
    return None


# ---------------------------------------------------------------------------
# Image pre-processing
# ---------------------------------------------------------------------------

def _preprocess(img_bytes: bytes):
    """Return a PIL Image optimised for Tesseract OCR."""
    from PIL import Image, ImageFilter, ImageOps
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Upscale small images — Tesseract works best at ~300 dpi / ~2000px wide
    w, h = img.size
    if w < 1200:
        scale = max(2, 1200 // w)
        img = img.resize((w * scale, h * scale), Image.LANCZOS)

    # Greyscale + mild sharpening
    grey = img.convert("L")
    sharp = grey.filter(ImageFilter.SHARPEN)
    return sharp


# ---------------------------------------------------------------------------
# Row / column reconstruction from bounding boxes
# ---------------------------------------------------------------------------

def _ocr_to_rows(img) -> List[List[dict]]:
    """
    Run pytesseract image_to_data and group word boxes into visual rows.
    Returns list-of-rows, each row = list of {text, x, y, w, h, conf}.
    """
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    data = pytesseract.image_to_data(
        img,
        config="--psm 6 --oem 3",
        output_type=pytesseract.Output.DICT,
    )

    words = []
    for i in range(len(data["text"])):
        txt = data["text"][i].strip()
        conf = int(data["conf"][i])
        if not txt or conf < 20:
            continue
        words.append({
            "text": txt,
            "x":    data["left"][i],
            "y":    data["top"][i],
            "w":    data["width"][i],
            "h":    data["height"][i],
            "conf": conf,
        })

    if not words:
        return []

    # Cluster by y-centre with tolerance = avg word height * 0.6
    avg_h = sum(w["h"] for w in words) / len(words)
    tol   = max(8, avg_h * 0.6)

    words_sorted = sorted(words, key=lambda w: w["y"])
    rows: List[List[dict]] = []
    current_row: List[dict] = [words_sorted[0]]

    for word in words_sorted[1:]:
        if abs(word["y"] - current_row[-1]["y"]) <= tol:
            current_row.append(word)
        else:
            rows.append(sorted(current_row, key=lambda w: w["x"]))
            current_row = [word]
    rows.append(sorted(current_row, key=lambda w: w["x"]))

    return rows


def _split_row(row: List[dict], label_boundary: int) -> Tuple[str, List[str]]:
    """
    Split a row's words into (label_text, [numeric_token, ...]).
    label_boundary is the x-coordinate that separates label from value columns.
    """
    label_words = [w["text"] for w in row if w["x"] < label_boundary]
    value_words = [w["text"] for w in row if w["x"] >= label_boundary]

    label = " ".join(label_words).strip()

    # Group consecutive value tokens by x-gap into distinct cell tokens
    if not value_words:
        return label, []

    # Rebuild value cells: words that are close together belong to one cell
    value_data = [w for w in row if w["x"] >= label_boundary]
    cells: List[str] = []
    cell_words: List[str] = [value_data[0]["text"]]

    for prev, curr in zip(value_data, value_data[1:]):
        gap = curr["x"] - (prev["x"] + prev["w"])
        if gap > prev["w"] * 1.2:  # large gap = new cell
            cells.append(" ".join(cell_words))
            cell_words = [curr["text"]]
        else:
            cell_words.append(curr["text"])
    cells.append(" ".join(cell_words))

    return label, cells


# ---------------------------------------------------------------------------
# Scale detection
# ---------------------------------------------------------------------------

def _detect_scale(all_text: str) -> Tuple[float, str]:
    t = all_text.lower()
    if re.search(r"\bcr\b|\bcrore|\bcrores\b", t):
        return 10_000_000, "Crores (Cr)"
    if re.search(r"\blakh|\blac\b|\blacs\b", t):
        return 100_000, "Lakhs"
    if re.search(r"\bmillion|\bmn\b", t):
        return 1_000_000, "Millions"
    if re.search(r"\bthousand", t):
        return 1_000, "Thousands"
    return 1.0, "Units (raw)"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _parse_by_lines(
    img,
    scale: float,
) -> Tuple[Dict[str, float], List[str]]:
    """
    PRIMARY parser: read OCR text line-by-line.

    More reliable than bounding-box approach for structured tables because
    it doesn't depend on x-coordinate clustering.  Each line of text is
    scanned for a field label followed by one or two numeric values.

    Tries PSM 6 (block) → PSM 4 (column) → PSM 11 (sparse) in order.
    Returns the result with the most fields extracted.
    """
    import pytesseract

    financials: Dict[str, float] = {}
    year_labels: List[str] = []
    best_result: Dict[str, float] = {}
    best_count = 0

    for psm in (6, 4, 11):
        config = f"--psm {psm} --oem 3"
        try:
            raw_text = pytesseract.image_to_string(img, config=config)
        except Exception as e:
            log.warning("PSM %d failed: %s", psm, e)
            continue

        result: Dict[str, float] = {}
        y_labels: List[str] = []

        for raw_line in raw_text.split("\n"):
            line = raw_line.strip()
            if not line:
                continue

            # Detect year-header row (e.g. "Financial year  2025  2024")
            y_hits = re.findall(r"\b(?:FY\s*\d{2,4}|\d{4}(?:-\d{2})?)\b", line, re.I)
            if y_hits:
                y_labels = y_hits
                log.debug("PSM%d year labels: %s", psm, y_labels)
                continue  # skip header row

            # Find the best matching field label in this line
            best_field: Optional[str] = None
            best_syn_len = 0
            line_lower = line.lower()

            # Exact match first
            for syn, field in _SYN_LOOKUP.items():
                if line_lower.startswith(syn) and len(syn) > best_syn_len:
                    best_field = field
                    best_syn_len = len(syn)

            # Contains match (only for synonyms ≥ 5 chars)
            if best_field is None:
                for syn, field in _SYN_LOOKUP.items():
                    if len(syn) >= 5 and syn in line_lower and len(syn) > best_syn_len:
                        best_field = field
                        best_syn_len = len(syn)

            if best_field is None:
                continue

            # Extract all numbers from the line
            nums = _extract_numbers_from_line(line)
            if not nums:
                continue

            cy_val = nums[0] * scale
            result[best_field] = cy_val
            log.debug("PSM%d  %-28s CY=%+.4g", psm, best_field, cy_val)

            if len(nums) >= 2:
                py_val = nums[1] * scale
                result[best_field + "_py"] = py_val
                log.debug("PSM%d  %-28s PY=%+.4g", psm, best_field + "_py", py_val)

        n = sum(1 for k in result if not k.endswith("_py"))
        log.info("PSM%d extracted %d CY fields", psm, n)
        if n > best_count:
            best_count = n
            best_result = result
            year_labels = y_labels

    if year_labels:
        best_result["financial_year"] = year_labels[0]
        if len(year_labels) >= 2:
            best_result["prior_financial_year"] = year_labels[1]

    return best_result, year_labels


def parse_image(raw_bytes: bytes, scale_override: Optional[float] = None) -> Tuple[dict, float, List[str]]:
    """
    Parse a financial table image using Tesseract OCR.

    Strategy:
      1. Pre-process image for OCR quality
      2. PRIMARY — line-by-line text extraction (PSM 6/4/11, takes best)
      3. FALLBACK — bounding-box column reconstruction (original approach)
      4. Merge results — prefer primary; fallback fills any gaps

    Returns:
        (financials_dict, confidence_0_to_1, warnings_list)
    """
    try:
        import pytesseract
    except ImportError:
        return {}, 0.0, [
            "pytesseract not installed. Run: pip install pytesseract "
            "and install Tesseract from https://github.com/UB-Mannheim/tesseract/wiki"
        ]

    pytesseract.pytesseract.tesseract_cmd = (
        r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )

    warnings: List[str] = []

    # ── 1. Pre-process ──────────────────────────────────────────────────────
    try:
        img = _preprocess(raw_bytes)
    except Exception as e:
        return {}, 0.0, [f"Image load failed: {e}"]

    # ── 2. Determine scale ──────────────────────────────────────────────────
    try:
        all_text_quick = pytesseract.image_to_string(img, config="--psm 6 --oem 3")
    except Exception:
        all_text_quick = ""

    if scale_override is not None and scale_override > 0:
        scale, scale_label = scale_override, f"user-override ×{scale_override:,.0f}"
    else:
        scale, scale_label = _detect_scale(all_text_quick)
    log.info("Scale: %s (×%g)", scale_label, scale)

    # ── 3. PRIMARY: line-based parser ────────────────────────────────────────
    financials, year_labels = _parse_by_lines(img, scale)

    # ── 4. FALLBACK: bounding-box parser for any still-missing fields ────────
    try:
        rows = _ocr_to_rows(img)
        if rows:
            img_width = img.size[0]
            # Detect label boundary from numeric word positions
            numeric_xs = [w["x"] for row in rows for w in row
                          if _parse_number(w["text"]) is not None]
            if numeric_xs:
                sorted_xs = sorted(numeric_xs)
                p10 = sorted_xs[max(0, int(len(sorted_xs) * 0.10) - 1)]
                label_boundary = max(int(img_width * 0.20), p10 - 10)
            else:
                label_boundary = int(img_width * 0.45)

            bb_year_labels: List[str] = []
            for row in rows:
                label_raw, cells = _split_row(row, label_boundary)
                y_hits = re.findall(r"\b(?:FY\s*\d{2,4}|\d{4}(?:-\d{2})?)\b",
                                    label_raw + " " + " ".join(cells), re.I)
                if y_hits:
                    bb_year_labels = y_hits
                    continue
                if not cells:
                    continue
                field = _match_label(label_raw)
                if field is None:
                    continue
                # Only fill gaps not already extracted by primary parser
                if field in financials:
                    continue
                nums = [n for c in cells for n in [_parse_number(c)] if n is not None]
                if not nums:
                    nums = _extract_numbers_from_line(" ".join(w["text"] for w in row))
                if nums:
                    financials[field] = nums[0] * scale
                    if len(nums) >= 2:
                        financials[field + "_py"] = nums[1] * scale
    except Exception as e:
        log.warning("Fallback box parser failed: %s", e)

    # ── 5. Build result ──────────────────────────────────────────────────────
    meta_keys = {"financial_year", "prior_financial_year", "company_name"}
    cy_fields = {k for k in financials if not k.endswith("_py") and k not in meta_keys}
    py_fields = {k for k in financials if k.endswith("_py")}
    n_cy, n_py = len(cy_fields), len(py_fields)

    log.info(
        "parse_image complete: %d CY fields, %d PY fields, scale=%s",
        n_cy, n_py, scale_label,
    )

    if n_cy == 0:
        warnings.append(
            "OCR could not extract any financial fields. "
            "The editable preview below is pre-populated with blank rows — "
            "type the values directly, or use Manual Entry in the sidebar."
        )
        confidence = 0.05
    else:
        confidence = min(0.90, 0.40 + n_cy * 0.06)
        if n_py > 0:
            warnings.append(
                f"Prior-year data extracted for {n_py} field(s) — trend analysis enabled."
            )
        else:
            warnings.append(
                "Only one year column detected. "
                "Enter prior-year values in the preview table to enable trend analysis."
            )

    return financials, confidence, warnings
