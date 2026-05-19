"""
Model 2: AI-Assisted Decision Engine.

Abstracts over Anthropic and OpenAI providers.
Returns a structured AIDecision with narrative, risk factors, confidence.
Falls back to a deterministic template if no API key is configured.
"""

from __future__ import annotations
import json
import os
import textwrap
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from config import (
    ANTHROPIC_API_KEY  as _CFG_ANTHROPIC_KEY,
    OPENAI_API_KEY     as _CFG_OPENAI_KEY,
    LLM_PROVIDER,
    LLM_MODEL,
)
from utils import fmt_currency, fmt_pct, fmt_ratio, get_logger, safe_divide

log = get_logger(__name__)


def _anthropic_key() -> str:
    """Return the Anthropic API key, checking os.environ at call-time.
    Allows a key entered at runtime (sidebar input) to be picked up immediately
    without restarting the app."""
    return os.environ.get("ANTHROPIC_API_KEY") or _CFG_ANTHROPIC_KEY


def _openai_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or _CFG_OPENAI_KEY


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class AIDecision:
    recommendation: str          # "APPROVE" | "CONDITIONAL" | "REJECT"
    confidence_level: str        # "High" | "Medium" | "Low"
    confidence_score: float      # 0.0 – 1.0
    risk_narrative: str          # 2-3 paragraph analysis
    key_risks: List[str]         # bullet points
    positive_factors: List[str]  # bullet points
    hidden_risks: List[str]      # less obvious concerns
    suggested_conditions: List[str]  # e.g. "Require audited statements"
    model_used: str


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_prompt(
    financials: dict,
    ratios: dict,
    scoring_summary: dict,
    external_summary: dict,
    payment_stats: dict,
) -> str:
    rev    = fmt_currency(financials.get("revenue", 0))
    ebitda = fmt_currency(financials.get("ebitda", 0))
    debt   = fmt_currency(financials.get("total_debt", 0))
    eq     = fmt_currency(financials.get("equity", 0))
    np_    = fmt_currency(financials.get("net_profit", 0))
    ocf    = fmt_currency(financials.get("operating_cash_flow", 0))

    ratio_lines = "\n".join(
        f"  - {r.get('name','').replace('_',' ').title()}: {r.get('display_value','N/A')} "
        f"(score {r.get('normalised_score',0):.0f}/100, signal: {r.get('signal','?')})"
        for r in ratios
    )

    # Build prior-year comparison block if available
    rev_py = financials.get("revenue_py")
    trend_lines = ""
    if rev_py:
        np_py   = financials.get("net_profit_py")
        debt_py = financials.get("total_debt_py")
        ocf_py  = financials.get("operating_cash_flow_py")
        trend_lines = "\nPRIOR-YEAR COMPARISON (YoY):\n"
        trend_lines += f"  Revenue PY:          {fmt_currency(rev_py)}\n"
        if np_py is not None:
            trend_lines += f"  Net Profit PY:       {fmt_currency(np_py)}\n"
        if debt_py is not None:
            trend_lines += f"  Total Debt PY:       {fmt_currency(debt_py)}\n"
        if ocf_py is not None:
            trend_lines += f"  Operating CF PY:     {fmt_currency(ocf_py)}\n"

    # Payment behavior — only include when data was actually uploaded
    has_payment = (payment_stats.get("total_invoices") or 0) > 0
    if has_payment:
        payment_section = textwrap.dedent(f"""
    PAYMENT BEHAVIOR (from uploaded invoice history):
      On-time payments:   {payment_stats.get('on_time_pct', 'N/A')}%
      Average days late:  {payment_stats.get('avg_days_late', 'N/A')} days
      Total invoices:     {payment_stats.get('total_invoices', 'N/A')}
        """).rstrip()
        payment_rule = (
            "- Incorporate payment behavior data into your assessment."
        )
    else:
        payment_section = (
            "\nPAYMENT BEHAVIOR: No payment history available for this client."
        )
        payment_rule = (
            "- Payment history is NOT available. Do NOT speculate about, reference, "
            "or mention payment behavior or invoice settlement in your analysis."
        )

    return textwrap.dedent(f"""
    You are a senior credit risk analyst at a B2B travel finance company.
    Evaluate the following corporate client for credit approval.

    === COMPANY: {financials.get('company_name', 'Unknown')} ===

    FINANCIAL SNAPSHOT:
      Revenue:              {rev}
      EBITDA:               {ebitda}
      Net Profit:           {np_}
      Total Debt:           {debt}
      Equity:               {eq}
      Operating Cash Flow:  {ocf}
    {trend_lines}
    FINANCIAL RATIOS (normalised score 0-100):
    {ratio_lines}

    COMPOSITE CREDIT SCORE: {scoring_summary.get('composite_score', 0):.1f} / 100
    RULE-BASED DECISION: {scoring_summary.get('risk_category', 'Unknown')}
    {payment_section}

    EXTERNAL RISK SIGNALS (last 90 days):
      Sentiment:   {external_summary.get('sentiment', 'N/A')}
      Risk flags:  {', '.join(external_summary.get('risk_flags', [])) or 'None'}
      News summary: {external_summary.get('summary', 'No data available')}

    ---

    Provide a structured credit assessment in the following JSON format:

    {{
      "recommendation": "APPROVE" | "CONDITIONAL" | "REJECT",
      "confidence_level": "High" | "Medium" | "Low",
      "confidence_score": 0.0-1.0,
      "risk_narrative": "2-3 paragraphs of holistic analysis",
      "key_risks": ["risk1", "risk2", ...],
      "positive_factors": ["factor1", "factor2", ...],
      "hidden_risks": ["hidden concern 1", ...],
      "suggested_conditions": ["condition if approving conditionally", ...]
    }}

    Rules:
    - Be specific. Reference actual numbers from the data.
    - Flag any contradictions between the rule-based score and qualitative signals.
    - For CONDITIONAL, list specific covenants or conditions.
    - hidden_risks should go beyond the obvious — look for structural, industry, or macro concerns.
    - risk_narrative MUST be formatted as 5–7 concise bullet points, each starting with "• ".
      Do NOT write prose paragraphs. Each bullet = one clear, data-referenced insight.
    - If prior-year data is available, include a bullet on the financial trajectory (improving/declining).
    {payment_rule}
    """).strip()


# ---------------------------------------------------------------------------
# Provider dispatch
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str) -> str:
    import anthropic
    client = anthropic.Anthropic(api_key=_anthropic_key())
    msg = client.messages.create(
        model=LLM_MODEL,
        max_tokens=2048,
        messages=[{"role": "user", "content": prompt}],
    )
    return msg.content[0].text


def _call_openai(prompt: str) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=_openai_key())
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


def _call_mock(
    scoring_summary: dict,
    external_summary: dict,
    has_payment: bool = False,
) -> str:
    """Deterministic mock response when no API key is configured."""
    score = scoring_summary.get("composite_score", 50)

    if score >= 70:
        rec  = "APPROVE"
        conf = 0.82
        narrative = (
            "• Solid financial fundamentals with adequate liquidity ratios and controlled leverage.\n"
            "• Revenue trends are stable and cash flow generation provides a comfortable buffer against debt.\n"
            + (
                "• Payment history indicates reliable settlement behaviour within agreed terms.\n"
                if has_payment else ""
            ) +
            "• EBITDA margins and net profitability are within acceptable benchmarks for the sector.\n"
            "• External signals are neutral-to-positive with no material risk flags identified.\n"
            "• Overall risk profile supports extending credit within recommended limits."
        )
        risks     = ["Moderate debt load warrants periodic monitoring", "Macro / sector sensitivity"]
        positives = ["Strong liquidity ratios", "Positive cash flow generation"] + (
            ["Consistent on-time payment history"] if has_payment else []
        )
        hidden    = ["Industry concentration risk", "FX exposure if operations are multinational"]
        conditions: list = []

    elif score >= 45:
        rec  = "CONDITIONAL"
        conf = 0.65
        narrative = (
            "• Mixed risk profile — revenue and profitability are acceptable but leverage is elevated.\n"
            "• Cash flow coverage sits below the preferred threshold, limiting repayment headroom.\n"
            + (
                "• Payment behaviour shows delays in the 30–60 day range — monitor closely.\n"
                if has_payment else ""
            ) +
            "• Working capital position is tight; current ratio approaching the 1.0 caution threshold.\n"
            "• External signals are broadly neutral with no critical flags at this time.\n"
            "• Credit may be extended on a conditional basis with a reduced initial limit and "
            "quarterly financial review."
        )
        risks     = ["Elevated debt-to-equity ratio", "Below-average cash flow coverage"] + (
            ["Payment delays in 30–60 day band"] if has_payment else []
        )
        positives = ["Positive EBITDA", "Revenue base is stable", "No critical external flags"]
        hidden    = ["Working capital squeeze under stress", "Potential covenant breach if margins compress"]
        conditions = ["Require audited financials", "Shortened net payment terms", "Quarterly financial review clause"]

    else:
        rec  = "REJECT"
        conf = 0.75
        narrative = (
            "• Critical liquidity shortfall — current and quick ratios are well below safe thresholds.\n"
            "• Excessive leverage with debt levels that appear unsustainable relative to earnings.\n"
            "• Profitability metrics are negative or near-zero; cash flow cannot service existing obligations.\n"
            + (
                "• Payment history shows systematic delays and patterns consistent with cash stress.\n"
                if has_payment else ""
            ) +
            "• External risk signals compound internal concerns — elevated caution warranted.\n"
            "• Extending credit at this time would represent imprudent exposure given the risk profile."
        )
        risks     = ["Critical liquidity shortfall", "Unsustainable leverage", "Negative or near-zero margins"] + (
            ["Poor payment track record"] if has_payment else []
        )
        positives = []
        hidden    = ["Going concern risk", "Potential restructuring or wind-down", "Industry headwinds amplifying stress"]
        conditions = []

    return json.dumps({
        "recommendation":       rec,
        "confidence_level":     "High" if conf > 0.75 else "Medium",
        "confidence_score":     conf,
        "risk_narrative":       narrative,
        "key_risks":            risks,
        "positive_factors":     positives,
        "hidden_risks":         hidden,
        "suggested_conditions": conditions,
    })


# ---------------------------------------------------------------------------
# JSON extraction (handles LLM responses with extra text)
# ---------------------------------------------------------------------------

def _sanitise_json_strings(text: str) -> str:
    """
    Walk the raw text character by character and escape any literal control
    characters (newline, carriage-return, tab) that appear inside JSON string
    values.  LLMs sometimes embed real newlines in multi-line string fields,
    which makes json.loads raise 'Unterminated string'.
    """
    result: list[str] = []
    in_string = False
    escape_next = False
    for ch in text:
        if escape_next:
            result.append(ch)
            escape_next = False
        elif ch == "\\" and in_string:
            result.append(ch)
            escape_next = True
        elif ch == '"':
            in_string = not in_string
            result.append(ch)
        elif in_string and ch == "\n":
            result.append("\\n")
        elif in_string and ch == "\r":
            result.append("\\r")
        elif in_string and ch == "\t":
            result.append("\\t")
        else:
            result.append(ch)
    return "".join(result)


def _extract_json(text: str) -> dict:
    import re
    # Strip markdown code fences if present
    text = re.sub(r"```(?:json)?", "", text).strip("` \n")
    # Find the first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    raw = match.group() if match else text

    # First attempt: direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Second attempt: escape literal control chars inside strings (LLM quirk)
    try:
        return json.loads(_sanitise_json_strings(raw))
    except json.JSONDecodeError:
        pass

    # Last resort: parse the full text as-is
    return json.loads(text)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_financials_from_image(
    image_bytes: bytes,
    media_type: str = "image/jpeg",
    company_name: str = "",
    scale_override: Optional[float] = None,
) -> tuple[dict, list[str]]:
    """
    Use Claude vision to extract BOTH current-year and prior-year financial data
    from a single image (annual reports always show two year columns side-by-side).

    Current-year fields use standard keys  (revenue, net_profit, …).
    Prior-year fields use _py suffix keys  (revenue_py, net_profit_py, …).

    Returns (financials_dict, warnings_list).
    All values are in raw INR — scale conversion (Cr/Lakh/Mn) is done inside the prompt.
    Falls back to empty dict + warning if no API key is configured.
    """
    import base64

    if not _anthropic_key():
        return {}, ["No ANTHROPIC_API_KEY — image extraction unavailable. Enter your key in the sidebar."]

    b64 = base64.standard_b64encode(image_bytes).decode()

    # ── Scale instruction ────────────────────────────────────────────────────
    _scale_names = {10_000_000: "Crores (Cr)", 100_000: "Lakhs",
                    1_000_000:  "Millions",    1_000:   "Thousands"}
    if scale_override and scale_override > 1:
        scale_name = _scale_names.get(int(scale_override), f"×{scale_override:,.0f}")
        scale_rule = (
            f"SCALE (user-confirmed): All values in this image are in {scale_name}. "
            f"Multiply every extracted number by {int(scale_override):,} to convert to raw INR "
            f"before placing it in the JSON. Do NOT auto-detect the scale."
        )
    else:
        scale_rule = textwrap.dedent("""
            SCALE DETECTION: Inspect the image for a scale indicator such as
            "₹ in Crores", "Figures in Lakhs", "₹ Mn", etc. and apply the rule:
              Crores / Cr  → multiply by 10,000,000
              Lakhs / Lacs → multiply by 100,000
              Millions / Mn→ multiply by 1,000,000
              Thousands    → multiply by 1,000
              No indicator → return the raw number as-is
        """).strip()

    prompt = textwrap.dedent(f"""
    You are a data extraction engine and financial analyst.

    ## OBJECTIVE
    Read the uploaded image containing financial data and extract every figure into
    the structured JSON format defined below.  The image shows a table with:
      - Column 1 : Metric label  (leftmost)
      - Column 2 : Current Year  (more recent — higher year number)
      - Column 3 : Prior Year    (older — lower year number)
    The first row of the table typically contains the year labels
    (e.g. "Financial year | 2025 | 2024").

    ## LABEL MAPPINGS  (common abbreviations in this format)
      "Current Liab."  → current_liabilities
      "Cash & Bank"    → cash_and_equivalents
      "Receivables"    → accounts_receivable
      "Payables"       → accounts_payable
      "Ops Cash Flow"  → operating_cash_flow
      "Net Worth"      → equity
      "Travel Spend"   → travel_cost

    ## EXTRACTION RULES
    1. Identify headers (year labels) and ALL data rows — do NOT skip any row.
    2. Normalize values:
       - Remove commas and currency symbols (₹, $, etc.)
       - Preserve negative values: "-3.71" stays -3.71; "(3.71)" becomes -3.71
       - Fix common OCR misreads: "l"→"1", "O"→"0", "I"→"1", "S"→"5", "B"→"8"
    3. Align every number to its correct metric row — do not shift values up or down.
    4. {scale_rule}
    5. If a value is genuinely unreadable, use null — do NOT guess.
    6. YoY validation (optional cross-check): verify that CY and PY numbers are
       in the same order of magnitude; flag obvious misalignments by setting to null.

    ## OUTPUT FORMAT
    Return ONLY a single valid JSON object using the exact keys below.
    Omit keys whose value is null or not present in the image.
    Do NOT include markdown fences, commentary, or explanation.

    {{
      "company_name":            "string — from header/footer if visible, else null",
      "financial_year":          "string — current year label e.g. 2025 or FY2025",
      "prior_financial_year":    "string — prior year label e.g. 2024 or FY2024",

      "revenue":                 <number — current year, converted to INR>,
      "ebitda":                  <number — current year, converted to INR>,
      "net_profit":              <number — current year, converted to INR>,
      "current_assets":          <number — current year, converted to INR>,
      "current_liabilities":     <number — current year, converted to INR>,
      "total_debt":              <number — current year, converted to INR>,
      "equity":                  <number — current year, converted to INR>,
      "cash_and_equivalents":    <number — current year, converted to INR>,
      "accounts_receivable":     <number — current year, converted to INR>,
      "accounts_payable":        <number — current year, converted to INR>,
      "operating_cash_flow":     <number — current year, converted to INR>,
      "interest_expense":        <number — current year, converted to INR>,
      "travel_cost":             <number — current year, only if travel/conveyance row present>,

      "revenue_py":              <number — prior year, converted to INR>,
      "ebitda_py":               <number — prior year, converted to INR>,
      "net_profit_py":           <number — prior year, converted to INR>,
      "current_assets_py":       <number — prior year, converted to INR>,
      "current_liabilities_py":  <number — prior year, converted to INR>,
      "total_debt_py":           <number — prior year, converted to INR>,
      "equity_py":               <number — prior year, converted to INR>,
      "cash_and_equivalents_py": <number — prior year, converted to INR>,
      "accounts_receivable_py":  <number — prior year, converted to INR>,
      "accounts_payable_py":     <number — prior year, converted to INR>,
      "operating_cash_flow_py":  <number — prior year, converted to INR>,
      "interest_expense_py":     <number — prior year, converted to INR>,
      "travel_cost_py":          <number — prior year, only if travel row present>
    }}
    """).strip()

    try:
        import anthropic
        client = anthropic.Anthropic(api_key=_anthropic_key())
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=1500,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": b64,
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            }],
        )
        raw = msg.content[0].text.strip()
        parsed = _extract_json(raw)

        # Inject company_name if not found in image but provided by user
        if company_name and not parsed.get("company_name"):
            parsed["company_name"] = company_name

        # Filter out null, "Check" sentinel, and non-numeric values
        extracted: dict = {}
        uncertain: list[str] = []
        for k, v in parsed.items():
            if v is None:
                continue
            if isinstance(v, str) and v.strip().lower() in ("check", "unclear", "n/a", "?"):
                uncertain.append(k)
                continue
            extracted[k] = v

        # Define expected field sets
        cy_fields = {
            "revenue", "ebitda", "net_profit", "current_assets", "current_liabilities",
            "total_debt", "equity", "cash_and_equivalents", "accounts_receivable",
            "accounts_payable", "operating_cash_flow", "interest_expense", "travel_cost",
        }
        py_fields = {f + "_py" for f in cy_fields}

        cy_extracted  = cy_fields & set(extracted.keys())
        cy_missing    = cy_fields - cy_extracted - {"travel_cost"}  # travel optional
        py_extracted_ = py_fields & set(extracted.keys())
        has_py        = bool(py_extracted_)

        warnings: list[str] = []

        # Uncertain / unreadable values
        if uncertain:
            warnings.append(
                f"⚠️ {len(uncertain)} value(s) marked unclear by Claude — shown as blank in "
                f"the preview for you to fill in: {', '.join(sorted(uncertain))}"
            )

        # Missing CY fields
        if cy_missing:
            warnings.append(
                f"Not found in image: {', '.join(sorted(cy_missing))}. "
                "Fill them in the editable preview below."
            )

        if has_py:
            warnings.append(
                f"Prior-year data extracted for {len(py_extracted_)} field(s) — "
                "trend analysis enabled."
            )
        else:
            warnings.append(
                "No prior-year column detected — trend score will be neutral. "
                "Add prior-year values in the preview table to enable trend analysis."
            )

        log.info(
            "Claude Vision extracted %d CY + %d PY fields  (uncertain=%d)",
            len(cy_extracted), len(py_extracted_), len(uncertain),
        )
        return extracted, warnings

    except Exception as exc:
        log.error("Image extraction failed: %s", exc, exc_info=True)
        return {}, [f"Image extraction error: {exc}"]


def run_ai_analysis(
    financials: dict,
    ratio_results: list,
    scoring_summary: dict,
    external_result: dict,
    payment_stats: dict,
) -> AIDecision:
    """
    Call the configured LLM provider and return a structured AIDecision.
    Gracefully degrades to mock if no key is available.
    """
    provider = LLM_PROVIDER.lower()

    # Determine effective provider
    if provider == "anthropic" and not _anthropic_key():
        log.warning("ANTHROPIC_API_KEY not set; falling back to mock AI response")
        provider = "mock"
    elif provider == "openai" and not _openai_key():
        log.warning("OPENAI_API_KEY not set; falling back to mock AI response")
        provider = "mock"

    # Payment stats might come from scoring sub-score detail
    if not payment_stats:
        payment_stats = {"on_time_pct": "N/A", "avg_days_late": "N/A", "total_invoices": 0}

    has_payment = (payment_stats.get("total_invoices") or 0) > 0

    prompt = _build_prompt(
        financials, ratio_results, scoring_summary, external_result, payment_stats
    )

    try:
        if provider == "anthropic":
            log.info("Calling Anthropic %s", LLM_MODEL)
            raw = _call_anthropic(prompt)
        elif provider == "openai":
            log.info("Calling OpenAI gpt-4o")
            raw = _call_openai(prompt)
        else:
            log.info("Using mock AI response (has_payment=%s)", has_payment)
            raw = _call_mock(scoring_summary, external_result, has_payment=has_payment)

        parsed = _extract_json(raw)

        return AIDecision(
            recommendation      = parsed.get("recommendation", "CONDITIONAL"),
            confidence_level    = parsed.get("confidence_level", "Medium"),
            confidence_score    = float(parsed.get("confidence_score", 0.6)),
            risk_narrative      = parsed.get("risk_narrative", ""),
            key_risks           = parsed.get("key_risks", []),
            positive_factors    = parsed.get("positive_factors", []),
            hidden_risks        = parsed.get("hidden_risks", []),
            suggested_conditions= parsed.get("suggested_conditions", []),
            model_used          = provider,
        )

    except Exception as exc:
        log.error("AI engine error: %s", exc, exc_info=True)
        # Graceful fallback: return mock based on score
        raw = _call_mock(scoring_summary, external_result, has_payment=has_payment)
        parsed = json.loads(raw)
        return AIDecision(
            recommendation      = parsed["recommendation"],
            confidence_level    = "Low",
            confidence_score    = 0.3,
            risk_narrative      = f"AI analysis failed ({exc}). Showing rule-based estimate only.",
            key_risks           = parsed["key_risks"],
            positive_factors    = parsed["positive_factors"],
            hidden_risks        = parsed["hidden_risks"],
            suggested_conditions= parsed["suggested_conditions"],
            model_used          = "fallback",
        )
