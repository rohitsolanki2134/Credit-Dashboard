"""
Central configuration: API keys, scoring weights, thresholds, constants.
All tuneable parameters live here so nothing is buried in business logic.
"""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import Dict, Tuple


# ---------------------------------------------------------------------------
# API / Integration keys (populated from environment or .env file)
# ---------------------------------------------------------------------------

ANTHROPIC_API_KEY: str = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
SERP_API_KEY: str = os.getenv("SERP_API_KEY", "")

# Google Cloud Vision (text-mode OCR provider)
GOOGLE_VISION_API_KEY: str = os.getenv("GOOGLE_VISION_API_KEY", "")

# Azure Computer Vision (text-mode OCR provider)
AZURE_VISION_ENDPOINT: str = os.getenv("AZURE_VISION_ENDPOINT", "")
AZURE_VISION_KEY: str = os.getenv("AZURE_VISION_KEY", "")

# Which LLM provider to use: "anthropic" | "openai" | "mock"
LLM_PROVIDER: str = os.getenv("LLM_PROVIDER", "anthropic")
LLM_MODEL: str = os.getenv("LLM_MODEL", "claude-sonnet-4-6")

# Lightweight model used for OCR text structuring (cheaper than the main model)
LLM_LITE_MODEL: str = os.getenv("LLM_LITE_MODEL", "claude-haiku-4-5")

# OCR provider for image-based financial extraction
# Options: "claude" | "openai" | "tesseract" | "google_vision" | "azure_vision"
OCR_PROVIDER: str = os.getenv("OCR_PROVIDER", "claude")

# ---------------------------------------------------------------------------
# Database
# ---------------------------------------------------------------------------

DB_PATH: str = os.getenv("DB_PATH", "credit_dashboard.db")

# ---------------------------------------------------------------------------
# Email / SMTP settings
# Populate SMTP_USER + SMTP_PASSWORD in your .env file.
# Gmail example : SMTP_HOST=smtp.gmail.com, SMTP_PORT=587
#                 Use an App Password (not your main password):
#                 Google Account → Security → 2-Step → App Passwords
# Outlook/O365  : SMTP_HOST=smtp.office365.com, SMTP_PORT=587
# ---------------------------------------------------------------------------

SMTP_HOST: str = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT: int = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER: str = os.getenv("SMTP_USER", "")
SMTP_PASSWORD: str = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM: str = os.getenv("SMTP_FROM", "credit-dashboard@travelplusapp.com")

DEFAULT_EMAIL_RECIPIENTS: list = [
    "rohit.solanki@travelplusapp.com",
    "sashank.kaushal@travelplusapp.com",
    "btcinvoice@travelplusapp.com",
]

# ---------------------------------------------------------------------------
# Scoring engine weights (must sum to 1.0)
# ---------------------------------------------------------------------------

SCORING_WEIGHTS: Dict[str, float] = {
    "financial_health":      0.35,   # was 0.40 — ceded 5% to trend
    "payment_behavior":      0.25,   # was 0.30 — ceded 5% to trend
    "external_risk":         0.15,   # was 0.20 — ceded 5% to trend
    "operational_stability": 0.10,
    "financial_trend":       0.15,   # YoY revenue / margin / debt trajectory
}

# ---------------------------------------------------------------------------
# Financial ratio benchmarks (industry-neutral defaults, override per sector)
# ---------------------------------------------------------------------------

@dataclass
class RatioBenchmark:
    """Min/ideal/max for a given ratio.  Scores are linearly interpolated."""
    low_risk_min: float   # >= this → full marks
    acceptable_min: float # >= this → partial marks
    critical_max: float   # >= this (for debt-like ratios) → zero / penalty
    higher_is_better: bool = True
    weight_in_financial: float = 1.0  # relative weight inside financial_health


RATIO_BENCHMARKS: Dict[str, RatioBenchmark] = {
    "current_ratio":       RatioBenchmark(low_risk_min=2.0,  acceptable_min=1.2, critical_max=0.8,  higher_is_better=True,  weight_in_financial=1.5),
    "quick_ratio":         RatioBenchmark(low_risk_min=1.5,  acceptable_min=0.8, critical_max=0.5,  higher_is_better=True,  weight_in_financial=1.5),
    "debt_to_equity":      RatioBenchmark(low_risk_min=0.5,  acceptable_min=1.5, critical_max=3.0,  higher_is_better=False, weight_in_financial=2.0),
    "interest_coverage":   RatioBenchmark(low_risk_min=5.0,  acceptable_min=2.0, critical_max=1.0,  higher_is_better=True,  weight_in_financial=2.0),
    "ebitda_margin":       RatioBenchmark(low_risk_min=0.20, acceptable_min=0.08, critical_max=0.0, higher_is_better=True,  weight_in_financial=1.5),
    "net_margin":          RatioBenchmark(low_risk_min=0.10, acceptable_min=0.03, critical_max=0.0, higher_is_better=True,  weight_in_financial=1.0),
    "cash_flow_coverage":  RatioBenchmark(low_risk_min=1.5,  acceptable_min=0.8, critical_max=0.3,  higher_is_better=True,  weight_in_financial=1.5),
    # DSO: lower is better — excellent <30d, ok 30-60d, warning 60-90d, critical >90d
    # Weight raised to 1.8 — collections quality directly predicts payment behaviour to us
    "dso":                 RatioBenchmark(low_risk_min=30,   acceptable_min=60,  critical_max=90,   higher_is_better=False, weight_in_financial=1.8),
}

# ---------------------------------------------------------------------------
# Risk category thresholds (composite 0-100 score)
# ---------------------------------------------------------------------------

RISK_THRESHOLDS: Dict[str, Tuple[float, float]] = {
    "Good":    (70.0, 100.0),
    "Average": (45.0,  69.9),
    "Risky":   (0.0,   44.9),
}

RISK_COLORS: Dict[str, str] = {
    "Good":    "#2ECC71",
    "Average": "#F39C12",
    "Risky":   "#E74C3C",
}

RISK_EMOJI: Dict[str, str] = {
    "Good":    "🟢",
    "Average": "🟡",
    "Risky":   "🔴",
}

# ---------------------------------------------------------------------------
# Credit limit engine parameters
# ---------------------------------------------------------------------------

CREDIT_LIMIT_BASE_PCT: float = 0.25          # kept for legacy; not used by main formula
CREDIT_LIMIT_MAX_MULTIPLIER: float = 1.5     # strong profiles can exceed base limit by 50%
CREDIT_LIMIT_MIN_MULTIPLIER: float = 0.0     # floor at 0 (reject)

# Minimum annual revenue (₹ Crores) for core credit eligibility
# Companies below this threshold receive a lower Operational Stability score
# and a risk flag; above it receives a positive signal.
MIN_REVENUE_CR: float = 200.0   # ₹ 200 Crores

# Travel-spend-based credit sizing
# Corporate travel typically runs 0.01–2.5% of revenue, scaling with company size:
# smaller companies spend a higher fraction on travel than large enterprises.
# When travel cost is not reported, it is estimated via a revenue-scaled formula
# (see credit_engine._estimate_travel_pct).
TRAVEL_BUDGET_PCT_DEFAULT: float = 0.025     # 2.5% — used only as a hard fallback
TRAVEL_BUDGET_PCT_MIN: float     = 0.0001    # 0.01% floor  (very large enterprises)
TRAVEL_BUDGET_PCT_MAX: float     = 0.025     # 2.5% ceiling (small companies)
TRAVEL_CREDIT_MONTHS: float      = 3.0       # base limit covers 3 months of travel spend

# Revenue anchors for the log-scale travel-pct curve (in ₹ Crores)
# At TRAVEL_SCALE_LOW_CR  → TRAVEL_BUDGET_PCT_MAX (2.5%)
# At TRAVEL_SCALE_HIGH_CR → TRAVEL_BUDGET_PCT_MIN (0.01%)
# In between: smooth logarithmic interpolation
TRAVEL_SCALE_LOW_CR:  float = 200.0       # ₹200 Cr  → 2.5% of revenue
TRAVEL_SCALE_HIGH_CR: float = 100_000.0   # ₹1 Lakh Cr → 0.01% of revenue

PAYMENT_TERMS_MAP: Dict[str, int] = {        # risk category → net days
    "Good":    45,
    "Average": 30,
    "Risky":   14,
}

# ---------------------------------------------------------------------------
# External risk engine
# ---------------------------------------------------------------------------

NEWS_LOOKBACK_DAYS: int = 365    # fetch up to 1 year; recency filter applied in code
NEWS_MAX_ARTICLES: int = 20

# Keywords for articles older than 1 year that remain credit-relevant (funding / M&A)
EVERGREEN_NEWS_KEYWORDS: list = [
    "funding", "funded", "raised", "raises", "series a", "series b", "series c",
    "series d", "pre-ipo", "ipo", "merger", "merges", "merged", "acquisition",
    "acquired", "acquires", "takeover", "buyout", "stake", "capital raise",
    "strategic investment", "private equity", "venture capital",
]

# Financial search terms appended to company-name queries (NewsAPI / SerpAPI / RSS)
FINANCIAL_SEARCH_TERMS: str = (
    "earnings OR revenue OR profit OR results OR funding OR merger OR acquisition "
    "OR debt OR bankruptcy OR IPO OR investment OR quarterly"
)

NEGATIVE_KEYWORDS = [
    "fraud", "lawsuit", "bankruptcy", "insolvency", "layoffs", "restructuring",
    "investigation", "penalty", "default", "downgrade", "scandal", "recall",
    "hack", "breach", "loss", "deficit", "miss", "warning", "regulatory",
    "debt trap", "npa", "write-off", "probe", "raid", "sebi notice",
    "enforcement directorate", "income tax raid", "cbi investigation",
]

POSITIVE_KEYWORDS = [
    "profit", "growth", "expansion", "acquisition", "partnership", "award",
    "record", "upgrade", "strong", "beat", "raise", "investment", "launch",
    "revenue growth", "margin improvement", "debt-free", "cash surplus",
    "order book", "contract win", "fundraise", "ipo",
]

# ---------------------------------------------------------------------------
# Parser confidence thresholds
# ---------------------------------------------------------------------------

CONFIDENCE_HIGH: float   = 0.80
CONFIDENCE_MEDIUM: float = 0.50
CONFIDENCE_LOW: float    = 0.0

# ---------------------------------------------------------------------------
# UI / display
# ---------------------------------------------------------------------------

APP_TITLE: str = "Credit Evaluation Dashboard"
APP_ICON: str  = "💳"
PAGE_LAYOUT: str = "wide"
