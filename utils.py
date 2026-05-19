"""
Shared utilities: logging, formatting, validation helpers.
"""

from __future__ import annotations
import logging
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def get_logger(name: str = "credit_dashboard") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler — force UTF-8 so ₹ and other symbols render on Windows
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (rotates daily by date suffix)
    log_file = LOG_DIR / f"credit_{datetime.now().strftime('%Y%m%d')}.log"
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


log = get_logger()


# ---------------------------------------------------------------------------
# Number / currency helpers
# ---------------------------------------------------------------------------

_CRORE = 10_000_000   # 1 Crore  = ₹ 1,00,00,000
_LAKH  =    100_000   # 1 Lakh   = ₹ 1,00,000


def fmt_currency(value: float, symbol: str = "₹") -> str:
    """Format a number in INR using Crore / Lakh / absolute notation."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    abs_val = abs(value)
    sign = "-" if value < 0 else ""
    if abs_val >= _CRORE:
        return f"{sign}{symbol}{abs_val / _CRORE:.2f} Cr"
    if abs_val >= _LAKH:
        return f"{sign}{symbol}{abs_val / _LAKH:.2f} L"
    return f"{sign}{symbol}{abs_val:,.0f}"


def fmt_pct(value: float, decimals: int = 1) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def fmt_ratio(value: float, decimals: int = 2) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value:.{decimals}f}x"


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """Division that never raises ZeroDivisionError."""
    if denominator == 0 or denominator is None:
        return default
    return numerator / denominator


# ---------------------------------------------------------------------------
# Text / extraction helpers
# ---------------------------------------------------------------------------

_MULTIPLIER_MAP = {
    # Western
    "thousand": 1_000, "thousands": 1_000,
    "million":  1_000_000, "millions": 1_000_000,
    "billion":  1_000_000_000, "billions": 1_000_000_000,
    # Indian
    "lakh":   100_000, "lakhs":   100_000, "lac":   100_000,
    "lacs":   100_000, "l":       100_000,
    "crore":  10_000_000, "crores": 10_000_000, "cr": 10_000_000,
    "k": 1_000,
}

# Matches optional sign, optional Rs./INR/₹ prefix, then a number
# (handles both Western 1,234,567 and Indian 12,34,567 comma styles)
_NUMBER_RE = re.compile(
    r"[-+]?"
    r"\(?"
    r"\s*(?:rs\.?|inr|₹)?\s*"
    r"([\d]{1,3}(?:[,\s][\d]{2,3})*(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"\s*\)?"
    r"(?:\s*(crores?|cr|lakhs?|lacs?|l\b|thousands?|millions?|billions?))?",
    re.IGNORECASE,
)


def extract_number(text: str) -> Optional[float]:
    """
    Pull the first monetary value from a string.
    Handles:
      - Indian format:  1,00,000 / 12,34,567 / Rs. 45.23 Cr / ₹120 L
      - Western format: 1,234,567 / $1.2M
      - Parenthetical negatives: (1,23,456) → negative
      - Suffixes: Cr / Crore / L / Lakh / Lac
    """
    if not text:
        return None
    # Strip currency symbols and normalise whitespace
    cleaned = re.sub(r"[$₹]", "", text).strip()
    m = _NUMBER_RE.search(cleaned)
    if not m:
        return None
    # Remove all commas (handles both Indian and Western grouping)
    raw = m.group(1).replace(",", "").replace(" ", "")
    try:
        value = float(raw)
    except ValueError:
        return None

    suffix = (m.group(2) or "").lower().rstrip(".")
    value *= _MULTIPLIER_MAP.get(suffix, 1)

    # Parenthetical → negative (Indian accounting format uses this for losses)
    window = text[max(0, m.start() - 3) : m.start() + 3]
    if "(" in window:
        value = -value

    return value


def clean_company_name(name: str) -> str:
    """Strip common legal suffixes and extra whitespace."""
    suffixes = r"\b(inc\.?|corp\.?|ltd\.?|llc\.?|plc\.?|limited|incorporated)\b"
    return re.sub(suffixes, "", name, flags=re.IGNORECASE).strip(" .,")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_financials(data: dict) -> tuple[bool, list[str]]:
    """
    Validate that extracted financial data has at minimum the critical fields.
    Returns (is_valid, list_of_warnings).
    """
    required = ["revenue", "total_debt", "equity", "current_assets", "current_liabilities"]
    warnings: list[str] = []

    for field in required:
        val = data.get(field)
        if val is None or (isinstance(val, float) and np.isnan(val)):
            warnings.append(f"Missing critical field: {field}")

    if data.get("revenue", 0) <= 0:
        warnings.append("Revenue is zero or negative — results may be unreliable")

    if data.get("equity", 1) < 0:
        warnings.append("Negative equity detected — company may be technically insolvent")

    return len(warnings) == 0, warnings


# ---------------------------------------------------------------------------
# Score → band helper
# ---------------------------------------------------------------------------

def score_to_band(score: float) -> tuple[str, str, str]:
    """Return (band_label, hex_color, emoji) for a 0–100 score."""
    from config import RISK_THRESHOLDS, RISK_COLORS, RISK_EMOJI
    for band, (lo, hi) in RISK_THRESHOLDS.items():
        if lo <= score <= hi:
            return band, RISK_COLORS[band], RISK_EMOJI[band]
    return "Risky", RISK_COLORS["Risky"], RISK_EMOJI["Risky"]


# ---------------------------------------------------------------------------
# Misc
# ---------------------------------------------------------------------------

def clamp(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, value))


def timestamp_now() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
