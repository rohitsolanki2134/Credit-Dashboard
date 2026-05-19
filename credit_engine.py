"""
Credit Limit Engine: computes dynamic, risk-adjusted credit exposure.

Formula:
  1. Travel Budget  = reported travel cost  OR  revenue × travel_budget_pct
  2. Base Limit     = (Travel Budget / 12) × TRAVEL_CREDIT_MONTHS  (3 months)
  3. Cash Flow Multiplier  — OCF/debt coverage
  4. Score Multiplier      — composite credit score
  5. Sentiment Multiplier  — external news signals
  6. DSO Multiplier        — receivables quality (high DSO = slow payer = risk)
  7. Final Limit    = Base × product(multipliers), clamped to [0, 2× Base]
  8. Max Safe Exposure = Final Limit × 1.25
  9. Payment Terms  = driven by risk band

The credit facility only needs to cover the customer's travel obligations to
the TMC/platform.  Anchoring to travel spend (1–4% of revenue) rather than
total revenue or OCF gives a limit that is proportional to the actual exposure
we would carry, regardless of company size.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

from config import (
    CREDIT_LIMIT_MAX_MULTIPLIER,
    CREDIT_LIMIT_MIN_MULTIPLIER,
    TRAVEL_BUDGET_PCT_DEFAULT,
    TRAVEL_BUDGET_PCT_MIN,
    TRAVEL_BUDGET_PCT_MAX,
    TRAVEL_CREDIT_MONTHS,
    TRAVEL_SCALE_LOW_CR,
    TRAVEL_SCALE_HIGH_CR,
    PAYMENT_TERMS_MAP,
)
import math

from utils import clamp, fmt_currency, fmt_pct, get_logger, safe_divide

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Revenue-scaled travel budget estimator
# ---------------------------------------------------------------------------

def _estimate_travel_pct(revenue: float) -> float:
    """
    Return the estimated travel spend as a fraction of revenue, scaled
    logarithmically with company size.

    Logic
    ─────
    Corporate travel as a % of revenue shrinks as companies grow:
      • Small  (≤ ₹200 Cr)     → 2.5%  (high-touch travel, smaller base)
      • Medium (₹1,000 Cr)     → ~1.1%
      • Large  (₹5,000 Cr)     → ~0.5%
      • Very large (₹50,000 Cr)→ ~0.07%
      • Enterprise (≥₹1L Cr)   → 0.01% (floor)

    Formula: log-linear interpolation between
      (TRAVEL_SCALE_LOW_CR, TRAVEL_BUDGET_PCT_MAX) and
      (TRAVEL_SCALE_HIGH_CR, TRAVEL_BUDGET_PCT_MIN)
    on a log–log scale so the curve is smooth across orders of magnitude.
    """
    if revenue <= 0:
        return TRAVEL_BUDGET_PCT_DEFAULT

    rev_cr = revenue / 10_000_000   # convert raw INR → Crores

    # Hard caps at both ends
    if rev_cr <= TRAVEL_SCALE_LOW_CR:
        return TRAVEL_BUDGET_PCT_MAX
    if rev_cr >= TRAVEL_SCALE_HIGH_CR:
        return TRAVEL_BUDGET_PCT_MIN

    # Logarithmic interpolation parameter t ∈ (0, 1)
    log_rev  = math.log(rev_cr)
    log_low  = math.log(TRAVEL_SCALE_LOW_CR)
    log_high = math.log(TRAVEL_SCALE_HIGH_CR)
    t = (log_rev - log_low) / (log_high - log_low)

    # Interpolate on log scale for the percentage too (powers-of-ten behaviour)
    log_pct_max = math.log(TRAVEL_BUDGET_PCT_MAX)
    log_pct_min = math.log(TRAVEL_BUDGET_PCT_MIN)
    pct = math.exp(log_pct_max + t * (log_pct_min - log_pct_max))

    return round(pct, 6)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class CreditDecision:
    approved: bool
    credit_limit: float
    max_safe_exposure: float
    payment_terms_days: int
    risk_category: str
    composite_score: float
    # Travel budget fields
    travel_budget: float          # annual travel spend used as credit anchor (₹)
    monthly_travel: float         # = travel_budget / 12
    travel_budget_source: str     # "reported" | "estimated (X.X% of revenue)"
    travel_budget_pct: float      # fraction of revenue (when estimated)
    # DSO
    dso_days: Optional[float]     # Days Sales Outstanding (None if no AR data)
    dso_multiplier: float
    # Multipliers
    base_limit: float
    monthly_revenue: float        # kept for reference / explainability
    cash_flow_multiplier: float
    score_multiplier: float
    sentiment_multiplier: float
    final_multiplier: float
    rationale: str
    conditions: list


# ---------------------------------------------------------------------------
# Multiplier helpers
# ---------------------------------------------------------------------------

def _cash_flow_multiplier(financials: dict) -> float:
    """
    OCF / Total Debt coverage → multiplier 0.50 – 1.20.
    Strong positive OCF earns a premium; negative OCF penalises.
    """
    ocf  = financials.get("operating_cash_flow") or 0
    debt = financials.get("total_debt") or 1
    if ocf < 0:
        return 0.55   # cash-burning — moderate reduction
    if debt == 0:
        return 1.20   # debt-free and cash-positive — excellent
    cov = safe_divide(ocf, debt)
    if cov >= 0.50: return 1.20
    if cov >= 0.30: return 1.05
    if cov >= 0.15: return 0.90
    if cov >= 0.05: return 0.72
    return 0.55


def _score_multiplier(composite_score: float) -> float:
    """
    Map 0–100 composite → multiplier 0.0–1.30.
    High scorers receive a premium multiplier to reward strong credit profiles.
    Below 35 = reject.
    """
    if composite_score >= 85: return 1.30
    if composite_score >= 70: return 1.10
    if composite_score >= 55: return 0.90
    if composite_score >= 45: return 0.68
    if composite_score >= 35: return 0.45
    return 0.00


def _sentiment_multiplier(sentiment: str, flag_count: int) -> float:
    """
    Positive sentiment earns a small premium; negative news reduces.
    Each risk flag adds a 4% reduction, capped at 15%.
    """
    base = {"Positive": 1.05, "Neutral": 1.00, "Negative": 0.85}.get(sentiment, 1.00)
    return max(0.70, base - min(flag_count * 0.04, 0.15))


def _dso_multiplier(dso_days: Optional[float]) -> float:
    """
    Receivables quality multiplier.
    No data → neutral (1.0).  Fast collectors earn a small premium.
      None      → 1.00 (no data = neutral)
      ≤30 days  → 1.05 (fast collector — slight uplift)
      ≤60 days  → 0.95 (normal)
      ≤90 days  → 0.80 (slow — caution)
      >90 days  → 0.65 (poor — significant reduction)
    """
    if dso_days is None:
        return 1.00
    if dso_days <= 30:
        return 1.05
    if dso_days <= 60:
        return 0.95
    if dso_days <= 90:
        return 0.80
    return 0.65


def _financial_stress_haircut(financials: dict) -> tuple[float, list[str]]:
    """
    Apply multiplicative penalties for structural financial weaknesses.
    Softened thresholds — penalties are meaningful but not overly punitive.
    Minimum combined haircut = 0.35.

    Returns (haircut_multiplier, list_of_reasons).
    """
    haircut = 1.0
    reasons = []

    equity  = financials.get("equity") or 0
    ocf     = financials.get("operating_cash_flow") or 0
    ca      = financials.get("current_assets") or 0
    cl      = financials.get("current_liabilities") or 1
    profit  = financials.get("net_profit") or 0
    debt    = financials.get("total_debt") or 0
    revenue = financials.get("revenue") or 1

    if equity < 0:
        haircut *= 0.75
        reasons.append(f"Negative equity ({fmt_currency(equity)}) → ×0.75")

    if ocf < 0:
        haircut *= 0.82
        reasons.append(f"Negative operating cash flow ({fmt_currency(ocf)}) → ×0.82")

    if profit < 0:
        haircut *= 0.90
        reasons.append(f"Net loss ({fmt_currency(profit)}) → ×0.90")

    current_ratio = safe_divide(ca, cl)
    if current_ratio < 1.0:
        haircut *= 0.85
        reasons.append(f"Current ratio {current_ratio:.2f} < 1.0 → ×0.85")

    debt_to_rev = safe_divide(debt, revenue)
    if debt_to_rev > 1.0:
        haircut *= 0.88
        reasons.append(f"Debt/Revenue {debt_to_rev:.2f}× > 1.0 → ×0.88")

    return max(0.35, haircut), reasons


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_credit_decision(
    financials: dict,
    scoring_result,
    external_result: dict,
    ai_recommendation: str = "CONDITIONAL",
) -> CreditDecision:
    # Unpack scoring — handle both dataclass and dict
    if hasattr(scoring_result, "composite_score"):
        composite = scoring_result.composite_score
        risk_cat  = scoring_result.risk_category
    else:
        composite = scoring_result.get("composite_score", 50)
        risk_cat  = scoring_result.get("risk_category", "Average")

    revenue     = financials.get("revenue") or 0
    monthly_rev = revenue / 12

    # --- Travel budget (credit anchor) ---
    travel_reported = financials.get("travel_cost") or 0

    if travel_reported > 0:
        travel_budget = travel_reported
        travel_pct    = safe_divide(travel_reported, revenue) if revenue > 0 else TRAVEL_BUDGET_PCT_DEFAULT
        travel_source = "reported from financial statements"
    else:
        # User may have supplied an override percentage via the sidebar slider
        user_pct = financials.get("travel_budget_pct")
        if user_pct and user_pct != TRAVEL_BUDGET_PCT_DEFAULT:
            # Explicit user override — honour it, but clamp to allowed range
            travel_pct    = clamp(user_pct, TRAVEL_BUDGET_PCT_MIN, TRAVEL_BUDGET_PCT_MAX)
            travel_source = f"user estimate ({travel_pct*100:.2g}% of revenue)"
        else:
            # Dynamic revenue-scaled estimate
            travel_pct    = _estimate_travel_pct(revenue)
            rev_cr        = revenue / 10_000_000
            travel_source = f"revenue-scaled estimate ({travel_pct*100:.2g}% of ₹{rev_cr:,.0f} Cr revenue)"
        travel_budget = revenue * travel_pct

    monthly_travel = travel_budget / 12
    base_limit     = monthly_travel * TRAVEL_CREDIT_MONTHS   # 3 months of travel spend

    # --- DSO ---
    accounts_recv = financials.get("accounts_receivable") or 0
    dso_days = safe_divide(accounts_recv * 365, revenue) if revenue > 0 and accounts_recv > 0 else None
    dso_mult  = _dso_multiplier(dso_days)

    # --- Other multipliers ---
    cf_mult   = _cash_flow_multiplier(financials)
    sc_mult   = _score_multiplier(composite)
    sentiment = external_result.get("sentiment", "Neutral")
    flags     = len(external_result.get("risk_flags", []))
    sent_mult = _sentiment_multiplier(sentiment, flags)

    # Financial stress haircut — penalises structural weakness
    stress_mult, stress_reasons = _financial_stress_haircut(financials)

    final_mult  = cf_mult * sc_mult * sent_mult * dso_mult * stress_mult
    final_mult  = clamp(final_mult, CREDIT_LIMIT_MIN_MULTIPLIER, CREDIT_LIMIT_MAX_MULTIPLIER)

    raw_limit    = base_limit * final_mult
    credit_limit = round(raw_limit / 5000) * 5000

    # Max safe exposure — conservative 10% buffer only
    max_exposure = credit_limit * 1.10

    # Payment terms
    payment_terms = PAYMENT_TERMS_MAP.get(risk_cat, 30)

    # Override: reject zone
    # Also treat limits below ₹25,000 as a practical reject — not meaningful
    MIN_MEANINGFUL_LIMIT = 25_000
    if ai_recommendation == "REJECT" or sc_mult == 0.0 or credit_limit < MIN_MEANINGFUL_LIMIT:
        credit_limit = 0.0
        max_exposure = 0.0
        approved     = False
    else:
        approved = credit_limit > 0

    rationale  = _build_rationale(
        composite, risk_cat, travel_budget, travel_source, monthly_travel,
        base_limit, cf_mult, sc_mult, sent_mult, dso_mult, dso_days,
        stress_mult, stress_reasons, final_mult, credit_limit,
        payment_terms, ai_recommendation,
    )
    conditions = _build_conditions(risk_cat, financials, composite, ai_recommendation)

    log.info(
        "Credit decision: approved=%s limit=%s terms=%d travel_budget=%s dso=%s",
        approved, fmt_currency(credit_limit), payment_terms,
        fmt_currency(travel_budget), f"{dso_days:.0f}d" if dso_days else "n/a",
    )

    return CreditDecision(
        approved             = approved,
        credit_limit         = credit_limit,
        max_safe_exposure    = max_exposure,
        payment_terms_days   = payment_terms,
        risk_category        = risk_cat,
        composite_score      = composite,
        travel_budget        = travel_budget,
        monthly_travel       = monthly_travel,
        travel_budget_source = travel_source,
        travel_budget_pct    = travel_pct,
        dso_days             = dso_days,
        dso_multiplier       = round(dso_mult, 3),
        base_limit           = base_limit,
        monthly_revenue      = monthly_rev,
        cash_flow_multiplier = round(cf_mult, 3),
        score_multiplier     = round(sc_mult, 3),
        sentiment_multiplier = round(sent_mult, 3),
        final_multiplier     = round(final_mult, 3),
        rationale            = rationale,
        conditions           = conditions,
    )


def _build_rationale(
    score, risk_cat, travel_budget, travel_source, monthly_travel,
    base_limit, cf_mult, sc_mult, sent_mult, dso_mult, dso_days,
    stress_mult, stress_reasons, final_mult, credit_limit, terms, ai_rec,
) -> str:
    dso_str = f"{dso_days:.0f} days" if dso_days is not None else "N/A"
    lines = [
        f"**Travel Budget** = {fmt_currency(travel_budget)}/yr ({travel_source})",
        f"**Monthly Travel** = {fmt_currency(monthly_travel)}",
        f"**Base Limit** = {fmt_currency(monthly_travel)} × {TRAVEL_CREDIT_MONTHS:.0f} months "
        f"= {fmt_currency(base_limit)}",
        "",
        f"**Cash Flow Multiplier**: {cf_mult:.2f}x — OCF/debt coverage",
        f"**Score Multiplier**: {sc_mult:.2f}x — composite score {score:.1f}/100 ({risk_cat})",
        f"**Sentiment Multiplier**: {sent_mult:.2f}x — external news",
        f"**DSO Multiplier**: {dso_mult:.2f}x — receivables {dso_str}",
        f"**Financial Stress Haircut**: {stress_mult:.2f}x",
    ]
    if stress_reasons:
        for r in stress_reasons:
            lines.append(f"  • {r}")
    lines += [
        "",
        f"**Recommended Credit Limit**: {fmt_currency(credit_limit)} | Net-{terms} terms",
        f"**AI Recommendation**: {ai_rec}",
    ]
    return "\n".join(lines)


def _build_conditions(risk_cat: str, financials: dict, score: float, ai_rec: str) -> list:
    conds = []
    if risk_cat == "Risky" or score < 50:
        conds.append("Monthly account review required")
        conds.append("Personal guarantee from directors recommended")
    if risk_cat == "Average":
        conds.append("Quarterly financial statement submission")
        conds.append("Credit line subject to annual renewal")
    if (financials.get("equity") or 0) < 0:
        conds.append("Negative equity — require audited financials before draw-down")
    dso = safe_divide(
        (financials.get("accounts_receivable") or 0) * 365,
        financials.get("revenue") or 1,
    )
    if dso > 90:
        conds.append(f"DSO {dso:.0f}d — provide debtor ageing report before draw-down")
    if ai_rec == "CONDITIONAL":
        conds.append("Subject to satisfactory completion of due diligence")
    if score >= 70:
        conds.append("Standard annual review applies")
    return conds


def credit_decision_to_dict(cd: CreditDecision) -> dict:
    return {
        "approved":             cd.approved,
        "credit_limit":         cd.credit_limit,
        "max_safe_exposure":    cd.max_safe_exposure,
        "payment_terms_days":   cd.payment_terms_days,
        "risk_category":        cd.risk_category,
        "composite_score":      cd.composite_score,
        "travel_budget":        cd.travel_budget,
        "monthly_travel":       cd.monthly_travel,
        "travel_budget_source": cd.travel_budget_source,
        "travel_budget_pct":    cd.travel_budget_pct,
        "dso_days":             cd.dso_days,
        "dso_multiplier":       cd.dso_multiplier,
        "base_limit":           cd.base_limit,
        "monthly_revenue":      cd.monthly_revenue,
        "cash_flow_multiplier": cd.cash_flow_multiplier,
        "score_multiplier":     cd.score_multiplier,
        "sentiment_multiplier": cd.sentiment_multiplier,
        "final_multiplier":     cd.final_multiplier,
        "rationale":            cd.rationale,
        "conditions":           cd.conditions,
    }
