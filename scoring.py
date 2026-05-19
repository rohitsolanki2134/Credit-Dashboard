"""
Model 1: Weighted Rule-Based Scoring Engine.

Pipeline:
  1. Compute financial ratios from raw financials
  2. Normalise each ratio to 0-100 using benchmark thresholds
  3. Aggregate into a Financial Health sub-score (40% of composite)
  4. Combine sub-scores: Financial Health + Payment Behavior +
     External Risk + Operational Stability → Composite 0-100
  5. Return full breakdown for dashboard explainability
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from config import (
    RATIO_BENCHMARKS,
    SCORING_WEIGHTS,
    RatioBenchmark,
    MIN_REVENUE_CR,
)
from utils import clamp, get_logger, safe_divide, score_to_band

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data classes — kept simple, JSON-serialisable
# ---------------------------------------------------------------------------

@dataclass
class RatioResult:
    name: str
    value: Optional[float]
    normalised_score: float       # 0-100
    benchmark_low_risk: float
    benchmark_acceptable: float
    higher_is_better: bool
    display_value: str
    signal: str                   # "good" | "ok" | "warning" | "critical"
    explanation: str


@dataclass
class SubScoreResult:
    name: str
    score: float                  # 0-100
    weight: float                 # e.g. 0.40
    weighted_contribution: float  # score × weight (already on 0-100 scale)
    detail: str


@dataclass
class ScoringResult:
    composite_score: float
    risk_category: str
    risk_color: str
    risk_emoji: str
    ratios: List[RatioResult]
    sub_scores: List[SubScoreResult]
    score_breakdown: Dict[str, float]   # pretty breakdown for display
    red_flags: List[str]
    positive_signals: List[str]


# ---------------------------------------------------------------------------
# Step 1 — Compute raw financial ratios
# ---------------------------------------------------------------------------

def compute_ratios(fin: dict) -> Dict[str, Optional[float]]:
    """
    Return dict of ratio_name → float (or None if data missing).
    All inputs expected in same currency unit (e.g. rupees).
    Includes DSO/DPO and YoY growth metrics when prior-year data present.
    """
    revenue           = fin.get("revenue")
    ebitda            = fin.get("ebitda")
    net_profit        = fin.get("net_profit")
    total_debt        = fin.get("total_debt")
    equity            = fin.get("equity")
    current_assets    = fin.get("current_assets")
    current_liab      = fin.get("current_liabilities")
    cash              = fin.get("cash_and_equivalents", 0) or 0
    inventory         = fin.get("inventory", 0) or 0
    op_cash_flow      = fin.get("operating_cash_flow")
    interest_expense  = fin.get("interest_expense")
    accounts_recv     = fin.get("accounts_receivable", 0) or 0
    accounts_pay      = fin.get("accounts_payable", 0) or 0
    total_assets      = fin.get("total_assets")

    ratios: Dict[str, Optional[float]] = {}

    # Liquidity
    ratios["current_ratio"] = (
        safe_divide(current_assets, current_liab) if current_assets is not None and current_liab else None
    )
    liquid_assets = (current_assets or 0) - inventory
    ratios["quick_ratio"] = (
        safe_divide(liquid_assets, current_liab) if current_assets is not None and current_liab else None
    )

    # Leverage
    ratios["debt_to_equity"] = (
        safe_divide(total_debt, equity) if total_debt is not None and equity else None
    )

    # Coverage
    if ebitda is not None and interest_expense is not None and interest_expense != 0:
        ratios["interest_coverage"] = safe_divide(ebitda, interest_expense)
    elif ebitda is not None and interest_expense == 0:
        ratios["interest_coverage"] = 999.0  # no interest → perfect coverage
    else:
        ratios["interest_coverage"] = None

    # Profitability
    ratios["ebitda_margin"] = (
        safe_divide(ebitda, revenue) if ebitda is not None and revenue else None
    )
    ratios["net_margin"] = (
        safe_divide(net_profit, revenue) if net_profit is not None and revenue else None
    )

    # Cash flow coverage (op_cash_flow / total_debt)
    ratios["cash_flow_coverage"] = (
        safe_divide(op_cash_flow, total_debt) if op_cash_flow is not None and total_debt else None
    )

    # DSO — Days Sales Outstanding (lower is better; high DSO = slow collections)
    if revenue and revenue > 0 and accounts_recv > 0:
        ratios["dso"] = safe_divide(accounts_recv * 365, revenue)
    else:
        ratios["dso"] = None

    # DPO — Days Payable Outstanding (stored for display, not scored)
    if revenue and revenue > 0 and accounts_pay > 0:
        ratios["_dpo"] = safe_divide(accounts_pay * 365, revenue)
    else:
        ratios["_dpo"] = None

    # YoY growth metrics (stored with _ prefix — used by trend scorer, not ratio scorer)
    revenue_py    = fin.get("revenue_py")
    net_profit_py = fin.get("net_profit_py")
    ebitda_py     = fin.get("ebitda_py")
    total_debt_py = fin.get("total_debt_py")
    equity_py     = fin.get("equity_py")

    ratios["_revenue_growth"]  = safe_divide(revenue - revenue_py, revenue_py) if revenue and revenue_py else None
    ratios["_margin_change"]   = (
        safe_divide(net_profit, revenue) - safe_divide(net_profit_py, revenue_py)
        if net_profit is not None and net_profit_py is not None and revenue and revenue_py else None
    )
    ratios["_debt_growth"]     = safe_divide(total_debt - total_debt_py, total_debt_py) if total_debt and total_debt_py else None

    # Reference values for display
    ratios["_revenue"]        = revenue
    ratios["_ebitda"]         = ebitda
    ratios["_net_profit"]     = net_profit
    ratios["_total_debt"]     = total_debt
    ratios["_equity"]         = equity
    ratios["_op_cash_flow"]   = op_cash_flow
    ratios["_accounts_recv"]  = accounts_recv if accounts_recv > 0 else None
    ratios["_accounts_pay"]   = accounts_pay if accounts_pay > 0 else None

    return ratios


# ---------------------------------------------------------------------------
# Step 2 — Normalise a single ratio to 0-100
# ---------------------------------------------------------------------------

def _normalise(value: float, bm: RatioBenchmark) -> float:
    """
    Piecewise linear interpolation:
      higher_is_better=True:
        value >= low_risk_min  → 100
        value >= acceptable_min → linear 40–99
        value >= critical_max  → linear 0–39
        value <  critical_max  → 0 (penalty zone)
      higher_is_better=False (e.g. debt-to-equity, lower is better):
        value <= low_risk_min   → 100   (low_risk_min is the ceiling threshold)
        value <= acceptable_min → 40–99
        value <= critical_max   → 0–39
        value >  critical_max   → 0
    """
    if bm.higher_is_better:
        lo, mid, hi = bm.critical_max, bm.acceptable_min, bm.low_risk_min
        if value >= hi:
            return 100.0
        if value >= mid:
            return 40.0 + 60.0 * (value - mid) / (hi - mid)
        if value >= lo:
            return 0.0 + 40.0 * (value - lo) / (mid - lo)
        return 0.0
    else:
        lo, mid, hi = bm.low_risk_min, bm.acceptable_min, bm.critical_max
        if value <= lo:
            return 100.0
        if value <= mid:
            return 40.0 + 60.0 * (mid - value) / (mid - lo)
        if value <= hi:
            return 0.0 + 40.0 * (hi - value) / (hi - mid)
        return 0.0


def _signal(score: float) -> str:
    if score >= 70:
        return "good"
    if score >= 40:
        return "ok"
    if score >= 20:
        return "warning"
    return "critical"


def _explanation(name: str, value: float, score: float, bm: RatioBenchmark) -> str:
    labels = {
        "current_ratio":      "Ability to meet short-term obligations",
        "quick_ratio":        "Liquid cover excluding inventory",
        "debt_to_equity":     "Financial leverage / balance-sheet risk",
        "interest_coverage":  "Earnings buffer over interest payments",
        "ebitda_margin":      "Core operating profitability",
        "net_margin":         "Bottom-line profitability",
        "cash_flow_coverage": "Cash generated relative to debt load",
        "dso":                "Days to collect receivables — lower means faster cash conversion",
    }
    base = labels.get(name, name)
    direction = "higher is better" if bm.higher_is_better else "lower is better"
    return f"{base} ({direction}). Benchmark (good): {bm.low_risk_min}"


def score_ratios(ratios: Dict[str, Optional[float]]) -> List[RatioResult]:
    from utils import fmt_ratio, fmt_pct

    results: List[RatioResult] = []
    for ratio_name, bm in RATIO_BENCHMARKS.items():
        value = ratios.get(ratio_name)
        if value is None:
            ns = 50.0  # neutral when data missing
            signal = "ok"
            display = "N/A"
        else:
            ns = clamp(_normalise(value, bm))
            signal = _signal(ns)
            if ratio_name in ("ebitda_margin", "net_margin"):
                display = fmt_pct(value)
            elif ratio_name == "dso":
                display = f"{value:.0f} days"
            else:
                display = fmt_ratio(value)

        results.append(RatioResult(
            name=ratio_name,
            value=value,
            normalised_score=round(ns, 1),
            benchmark_low_risk=bm.low_risk_min,
            benchmark_acceptable=bm.acceptable_min,
            higher_is_better=bm.higher_is_better,
            display_value=display,
            signal=signal,
            explanation=_explanation(ratio_name, value or 0, ns, bm),
        ))
    return results


# ---------------------------------------------------------------------------
# Step 3 — Financial Health sub-score
# ---------------------------------------------------------------------------

def financial_health_score(ratio_results: List[RatioResult]) -> float:
    """Weighted average of normalised ratio scores."""
    total_weight = 0.0
    weighted_sum = 0.0
    for rr in ratio_results:
        bm = RATIO_BENCHMARKS[rr.name]
        w = bm.weight_in_financial
        weighted_sum += rr.normalised_score * w
        total_weight += w
    if total_weight == 0:
        return 50.0
    return clamp(weighted_sum / total_weight)


# ---------------------------------------------------------------------------
# Step 4 — Payment Behavior sub-score
# ---------------------------------------------------------------------------

def payment_behavior_score(payment_records: list) -> Tuple[float, dict]:
    """
    Score derived from historical payment records.
    Returns (score_0_100, stats_dict).
    """
    if not payment_records:
        return 50.0, {"note": "No payment history — defaulting to neutral 50"}

    total = len(payment_records)
    buckets = {"ontime": 0, "late_30": 0, "late_60": 0, "late_90": 0, "default": 0, "unknown": 0}
    for r in payment_records:
        status = r.get("status", "unknown")
        buckets[status] = buckets.get(status, 0) + 1

    pct_ontime  = buckets["ontime"]  / total
    pct_late30  = buckets["late_30"] / total
    pct_late60  = buckets["late_60"] / total
    pct_late90  = buckets["late_90"] / total
    pct_default = buckets["default"] / total

    # Weighted penalty model
    score = (
        pct_ontime  * 100 +
        pct_late30  *  70 +
        pct_late60  *  40 +
        pct_late90  *  15 +
        pct_default *   0
    )
    score = clamp(score)

    avg_days_late = np.mean(
        [r.get("days_late", 0) or 0 for r in payment_records]
    )

    stats = {
        "total_invoices": total,
        "on_time_pct":    round(pct_ontime * 100, 1),
        "late_30_pct":    round(pct_late30 * 100, 1),
        "late_60_pct":    round(pct_late60 * 100, 1),
        "late_90_pct":    round(pct_late90 * 100, 1),
        "default_pct":    round(pct_default * 100, 1),
        "avg_days_late":  round(float(avg_days_late), 1),
        "buckets":        buckets,
    }
    return clamp(score), stats


# ---------------------------------------------------------------------------
# Step 5 — External Risk sub-score
# ---------------------------------------------------------------------------

def external_risk_score(external_result: dict) -> float:
    """
    Convert external risk engine output to 0–100.
    Positive sentiment → high score; negative → low.
    """
    sentiment = external_result.get("sentiment", "Neutral")
    risk_flag_count = len(external_result.get("risk_flags", []))

    base = {"Positive": 80, "Neutral": 60, "Negative": 25}.get(sentiment, 60)
    penalty = min(risk_flag_count * 8, 40)  # up to 40pt penalty
    return clamp(base - penalty)


# ---------------------------------------------------------------------------
# Step 6 — Operational Stability sub-score (qualitative proxy)
# ---------------------------------------------------------------------------

def operational_stability_score(fin: dict, external: dict) -> Tuple[float, str]:
    """
    Heuristic score combining three components:
      1. Revenue scale vs. ₹200 Cr eligibility threshold  (primary — up to ±30 pts)
      2. Cash buffer adequacy                              (up to +15 pts)
      3. External risk flag penalty                        (−5 pts per flag)

    Returns (score_0_100, detail_string).
    """
    revenue    = fin.get("revenue", 0) or 0
    revenue_cr = revenue / 10_000_000   # convert raw INR → Crores

    # ── 1. Turnover size relative to ₹200 Cr threshold ──────────────────────
    if revenue_cr >= 500:
        size_score = 30
        size_lbl   = f"₹{revenue_cr:,.0f} Cr — large enterprise, well above threshold"
    elif revenue_cr >= MIN_REVENUE_CR:
        size_score = 20
        size_lbl   = f"₹{revenue_cr:,.0f} Cr — above ₹{MIN_REVENUE_CR:.0f} Cr threshold ✓"
    elif revenue_cr >= 100:
        size_score = 5
        size_lbl   = f"₹{revenue_cr:,.0f} Cr — below ₹{MIN_REVENUE_CR:.0f} Cr threshold"
    elif revenue_cr >= 50:
        size_score = -5
        size_lbl   = f"₹{revenue_cr:,.0f} Cr — well below ₹{MIN_REVENUE_CR:.0f} Cr threshold"
    elif revenue_cr > 0:
        size_score = -15
        size_lbl   = f"₹{revenue_cr:,.0f} Cr — too small for standard credit product"
    else:
        size_score = -20
        size_lbl   = "Revenue not available"

    # ── 2. Cash buffer ───────────────────────────────────────────────────────
    cash       = fin.get("cash_and_equivalents", 0) or 0
    monthly_rev = revenue / 12
    if monthly_rev > 0 and cash >= monthly_rev * 3:
        cash_score = 15
        cash_lbl   = "Cash ≥ 3 months revenue (strong buffer)"
    elif monthly_rev > 0 and cash >= monthly_rev:
        cash_score = 8
        cash_lbl   = "Cash ≥ 1 month revenue (adequate)"
    else:
        cash_score = 0
        cash_lbl   = ""

    # ── 3. External flags penalty ────────────────────────────────────────────
    flags      = len(external.get("risk_flags", []))
    flag_score = -(flags * 5)

    total = clamp(50.0 + size_score + cash_score + flag_score)
    parts = [size_lbl]
    if cash_lbl:
        parts.append(cash_lbl)
    if flags:
        parts.append(f"{flags} external risk flag(s)")
    return total, " | ".join(parts)


# ---------------------------------------------------------------------------
# Step 7 — Financial Trend sub-score (requires prior-year data)
# ---------------------------------------------------------------------------

def financial_trend_score(fin: dict) -> Tuple[float, str]:
    """
    Score the company's financial trajectory using YoY comparisons.
    Returns (score_0_100, detail_string).

    Uses all available prior-year fields — not just revenue.  Each metric
    that has a CY+PY pair contributes a weighted component; missing pairs
    are simply skipped.  Only falls back to neutral 50 when NO prior-year
    data at all is present.

    Component weights (always normalise to 100%):
      Revenue growth      40%
      Net margin change   25%
      OCF trend           20%   (falls back to EBITDA margin if OCF absent)
      Debt trajectory     15%
    """
    revenue_cy = fin.get("revenue") or 0
    revenue_py = fin.get("revenue_py") or 0
    np_cy      = fin.get("net_profit") or 0
    np_py      = fin.get("net_profit_py") or 0
    ebitda_cy  = fin.get("ebitda") or 0
    ebitda_py  = fin.get("ebitda_py") or 0
    debt_cy    = fin.get("total_debt") or 0
    debt_py    = fin.get("total_debt_py") or 0
    ocf_cy     = fin.get("operating_cash_flow") or 0
    ocf_py     = fin.get("operating_cash_flow_py") or 0

    # Check whether ANY prior-year figure was provided
    has_any_py = any([revenue_py, np_py, ebitda_py, debt_py, ocf_py,
                      fin.get("equity_py"), fin.get("current_assets_py")])
    if not has_any_py:
        return 50.0, "No prior-year data — trend score neutral"

    # ── Component scores (None = data not available, skip) ──────────────────
    components: List[Tuple[float, float, str]] = []  # (score, weight, label)

    # --- Revenue growth (weight 40) ---
    if revenue_py > 0 and revenue_cy > 0:
        rev_g = safe_divide(revenue_cy - revenue_py, revenue_py)
        if rev_g >= 0.20:
            s, lbl = 100, f"Revenue +{rev_g*100:.1f}% (strong)"
        elif rev_g >= 0.10:
            s, lbl = 80,  f"Revenue +{rev_g*100:.1f}% (healthy)"
        elif rev_g >= 0.03:
            s, lbl = 65,  f"Revenue +{rev_g*100:.1f}% (modest)"
        elif rev_g >= -0.05:
            s, lbl = 50,  f"Revenue {rev_g*100:+.1f}% (flat)"
        elif rev_g >= -0.15:
            s, lbl = 30,  f"Revenue {rev_g*100:.1f}% (declining)"
        else:
            s, lbl = 10,  f"Revenue {rev_g*100:.1f}% (sharp fall)"
        components.append((s, 40, lbl))

    # --- Net margin change (weight 25) — needs revenue CY & PY ---
    if revenue_cy > 0 and revenue_py > 0:
        m_cy = safe_divide(np_cy, revenue_cy)
        m_py = safe_divide(np_py, revenue_py)
        delta = m_cy - m_py
        if delta >= 0.03:
            s, lbl = 100, f"Margin +{delta*100:.1f}pp (improving)"
        elif delta >= 0.00:
            s, lbl = 70,  f"Margin {delta*100:+.1f}pp (stable)"
        elif delta >= -0.03:
            s, lbl = 45,  f"Margin {delta*100:.1f}pp (slipping)"
        else:
            s, lbl = 15,  f"Margin {delta*100:.1f}pp (deteriorating)"
        components.append((s, 25, lbl))
    elif ebitda_py != 0 and revenue_py > 0 and revenue_cy > 0:
        # Fallback: EBITDA margin when net_profit_py not available
        em_cy = safe_divide(ebitda_cy, revenue_cy)
        em_py = safe_divide(ebitda_py, revenue_py)
        delta = em_cy - em_py
        if delta >= 0.03:
            s, lbl = 100, f"EBITDA margin +{delta*100:.1f}pp (improving)"
        elif delta >= 0.00:
            s, lbl = 70,  f"EBITDA margin {delta*100:+.1f}pp (stable)"
        elif delta >= -0.03:
            s, lbl = 45,  f"EBITDA margin {delta*100:.1f}pp (slipping)"
        else:
            s, lbl = 15,  f"EBITDA margin {delta*100:.1f}pp (deteriorating)"
        components.append((s, 25, lbl))

    # --- Operating cash flow trend (weight 20) ---
    if ocf_py != 0:
        if ocf_cy >= 0 and ocf_py > 0:
            ocf_g = safe_divide(ocf_cy - ocf_py, abs(ocf_py))
            if ocf_g >= 0.20:
                s, lbl = 100, f"OCF +{ocf_g*100:.1f}% (strong)"
            elif ocf_g >= 0.05:
                s, lbl = 80,  f"OCF +{ocf_g*100:.1f}% (growing)"
            elif ocf_g >= -0.10:
                s, lbl = 60,  f"OCF {ocf_g*100:+.1f}% (stable)"
            elif ocf_g >= -0.25:
                s, lbl = 35,  f"OCF {ocf_g*100:.1f}% (weakening)"
            else:
                s, lbl = 10,  f"OCF {ocf_g*100:.1f}% (deteriorating)"
        elif ocf_cy >= 0 and ocf_py < 0:
            s, lbl = 90, "OCF turned positive (recovery)"
        elif ocf_cy < 0 and ocf_py >= 0:
            s, lbl = 10, "OCF turned negative (deteriorating)"
        else:
            # Both negative — deeper negative is worse
            s = 20 if ocf_cy < ocf_py else 40
            lbl = "OCF negative both years"
        components.append((s, 20, lbl))

    # --- Debt trajectory (weight 15) ---
    if debt_py > 0:
        dbt_g = safe_divide(debt_cy - debt_py, debt_py)
        if dbt_g <= -0.10:
            s, lbl = 100, f"Debt −{abs(dbt_g)*100:.1f}% (reducing)"
        elif dbt_g <= 0.05:
            s, lbl = 75,  f"Debt {dbt_g*100:+.1f}% (stable)"
        elif dbt_g <= 0.20:
            s, lbl = 50,  f"Debt +{dbt_g*100:.1f}% (rising)"
        else:
            s, lbl = 20,  f"Debt +{dbt_g*100:.1f}% (surging)"
        components.append((s, 15, lbl))
    elif debt_cy == 0 and debt_py == 0:
        components.append((80, 15, "Debt-free both years"))

    if not components:
        return 50.0, "Insufficient comparable data for trend — neutral"

    # Normalise weights to sum to 100
    total_w = sum(w for _, w, _ in components)
    composite = sum(s * w / total_w for s, w, _ in components)
    detail = " | ".join(lbl for _, _, lbl in components)
    return clamp(composite), detail


# ---------------------------------------------------------------------------
# Dynamic weight redistribution
# ---------------------------------------------------------------------------

def _effective_weights(has_payment: bool) -> Dict[str, float]:
    """
    Return the scoring weight dict adjusted for data availability.

    When payment history is absent the 25% "payment_behavior" weight is
    redistributed proportionally among the other four sub-scores so the
    composite always sums to 100%.

    Example redistribution (no payment history):
      financial_health:      0.35 → 0.467
      external_risk:         0.15 → 0.200
      operational_stability: 0.10 → 0.133
      financial_trend:       0.15 → 0.200
    """
    w = dict(SCORING_WEIGHTS)
    if not has_payment:
        freed = w.pop("payment_behavior")
        total_remaining = sum(w.values())
        for k in list(w):
            w[k] = w[k] + freed * (w[k] / total_remaining)
    return w


# ---------------------------------------------------------------------------
# Master scoring function
# ---------------------------------------------------------------------------

def run_scoring(
    financials: dict,
    payment_records: list,
    external_result: dict,
) -> ScoringResult:
    """
    Run the full dual-layer weighted scoring pipeline.
    Returns a ScoringResult with full explainability breakdown.
    """
    log.info("Running scoring pipeline for %s", financials.get("company_name", "unknown"))

    has_payment = bool(payment_records)
    eff_w = _effective_weights(has_payment)

    # --- Ratios ---
    ratios_raw     = compute_ratios(financials)
    ratio_results  = score_ratios(ratios_raw)

    # --- Sub-scores ---
    fh_score  = financial_health_score(ratio_results)
    pb_score, pb_stats = payment_behavior_score(payment_records)
    er_score  = external_risk_score(external_result)
    os_score, os_detail = operational_stability_score(financials, external_result)
    ft_score, ft_detail = financial_trend_score(financials)

    pb_weight = eff_w.get("payment_behavior", 0.0)
    pb_detail = (
        f"{pb_stats.get('total_invoices', 0)} invoices analysed"
        if has_payment
        else "No payment history — weight redistributed to other parameters"
    )

    sub_scores = [
        SubScoreResult("Financial Health",
                       fh_score, eff_w["financial_health"],
                       fh_score * eff_w["financial_health"],
                       "8 financial ratios incl. DSO, liquidity, leverage, profitability"),
        SubScoreResult("Payment Behavior",
                       pb_score, pb_weight,
                       pb_score * pb_weight,
                       pb_detail),
        SubScoreResult("External Risk",
                       er_score, eff_w["external_risk"],
                       er_score * eff_w["external_risk"],
                       f"News sentiment: {external_result.get('sentiment', 'N/A')}"),
        SubScoreResult("Operational Stability",
                       os_score, eff_w["operational_stability"],
                       os_score * eff_w["operational_stability"],
                       os_detail),
        SubScoreResult("Financial Trend",
                       ft_score, eff_w["financial_trend"],
                       ft_score * eff_w["financial_trend"],
                       ft_detail),
    ]

    # --- Composite ---
    composite = sum(ss.weighted_contribution for ss in sub_scores)
    composite = clamp(composite)

    band, color, emoji = score_to_band(composite)

    # --- Red flags ---
    red_flags: List[str] = []
    positive_signals: List[str] = []

    for rr in ratio_results:
        if rr.signal == "critical":
            red_flags.append(f"{rr.name.replace('_',' ').title()}: {rr.display_value} (critical)")
        elif rr.signal == "good":
            positive_signals.append(f"{rr.name.replace('_',' ').title()}: {rr.display_value}")

    if has_payment:
        if pb_stats.get("default_pct", 0) > 5:
            red_flags.append(f"Payment defaults: {pb_stats['default_pct']}% of invoices")
        if pb_stats.get("avg_days_late", 0) > 45:
            red_flags.append(f"Average payment delay: {pb_stats['avg_days_late']:.0f} days")
    for flag in external_result.get("risk_flags", []):
        red_flags.append(f"External: {flag}")

    # Build score_breakdown with actual effective weights shown
    def _pct(key: str) -> str:
        return f"{eff_w.get(key, 0) * 100:.0f}%"

    score_breakdown = {
        f"Financial Health ({_pct('financial_health')})":           round(fh_score * eff_w["financial_health"], 1),
        f"Payment Behavior ({_pct('payment_behavior')})":           round(pb_score * pb_weight, 1),
        f"External Risk ({_pct('external_risk')})":                 round(er_score * eff_w["external_risk"], 1),
        f"Operational Stability ({_pct('operational_stability')})": round(os_score * eff_w["operational_stability"], 1),
        f"Financial Trend ({_pct('financial_trend')})":             round(ft_score * eff_w["financial_trend"], 1),
        "Composite Score":                                          round(composite, 1),
    }

    # Revenue threshold flag (₹200 Cr eligibility)
    revenue_cr = (financials.get("revenue") or 0) / 10_000_000
    if revenue_cr > 0:
        if revenue_cr < MIN_REVENUE_CR:
            red_flags.append(
                f"Revenue ₹{revenue_cr:,.0f} Cr is below ₹{MIN_REVENUE_CR:.0f} Cr "
                "preferred eligibility threshold"
            )
        elif revenue_cr >= 500:
            positive_signals.append(
                f"Revenue ₹{revenue_cr:,.0f} Cr — large enterprise, well above ₹{MIN_REVENUE_CR:.0f} Cr threshold"
            )
        else:
            positive_signals.append(
                f"Revenue ₹{revenue_cr:,.0f} Cr — above ₹{MIN_REVENUE_CR:.0f} Cr eligibility threshold ✓"
            )

    # Trend-specific flags
    if financials.get("revenue_py"):
        rev_g = ratios_raw.get("_revenue_growth")
        if rev_g is not None and rev_g < -0.10:
            red_flags.append(f"Revenue declining YoY: {rev_g*100:.1f}%")
        elif rev_g is not None and rev_g >= 0.15:
            positive_signals.append(f"Strong revenue growth YoY: +{rev_g*100:.1f}%")
        dso_val = ratios_raw.get("dso")
        if dso_val is not None and dso_val > 90:
            red_flags.append(f"DSO critically high: {dso_val:.0f} days (collections risk)")

    result = ScoringResult(
        composite_score=round(composite, 1),
        risk_category=band,
        risk_color=color,
        risk_emoji=emoji,
        ratios=ratio_results,
        sub_scores=sub_scores,
        score_breakdown=score_breakdown,
        red_flags=red_flags,
        positive_signals=positive_signals,
    )

    log.info("Scoring complete: composite=%.1f band=%s flags=%d", composite, band, len(red_flags))
    return result


def scoring_result_to_dict(sr: ScoringResult) -> dict:
    """Convert ScoringResult to a plain dict for JSON serialisation."""
    d = {
        "composite_score":  sr.composite_score,
        "risk_category":    sr.risk_category,
        "risk_color":       sr.risk_color,
        "risk_emoji":       sr.risk_emoji,
        "score_breakdown":  sr.score_breakdown,
        "red_flags":        sr.red_flags,
        "positive_signals": sr.positive_signals,
        "sub_scores": [asdict(ss) for ss in sr.sub_scores],
        "ratios":     [asdict(rr) for rr in sr.ratios],
    }
    return d
