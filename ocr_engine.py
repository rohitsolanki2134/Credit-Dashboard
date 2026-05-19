"""
Pluggable OCR Engine for financial document image extraction.

Architecture
────────────
Two provider modes:

  vision   – Provider receives image bytes and returns structured financial
             JSON directly (one LLM call). Best accuracy.
             Built-in: claude, openai

  text     – Provider returns raw OCR text from the image. A shared
             LLM-lite parse step then structures it into financial fields.
             Built-in: tesseract, google_vision, azure_vision

Adding a custom provider
────────────────────────
    from ocr_engine import register_ocr_provider

    @register_ocr_provider(
        "my_ocr",
        mode        = "text",        # "text" | "vision"
        label       = "My OCR Tool",
        description = "Company XYZ cloud OCR API",
    )
    def _my_provider(image_bytes: bytes, media_type: str, **_) -> str:
        # Call your OCR API, return raw text transcription
        return raw_text

That's it — the provider appears automatically in the UI selector and in
list_providers(), with availability detection based on whether the function
raises ImportError / missing-key exceptions.

Public API
──────────
    available = list_providers()
    financials, warnings = extract_financials(
        image_bytes, media_type,
        provider      = "claude",
        company_name  = "Acme Corp",
        scale_override= 10_000_000,
    )
"""

from __future__ import annotations

import base64
import io
import json
import os
import textwrap
from typing import Any, Callable, Dict, List, Optional, Tuple

from config import (
    ANTHROPIC_API_KEY as _CFG_ANT_KEY,
    OPENAI_API_KEY    as _CFG_OAI_KEY,
    GOOGLE_VISION_API_KEY,
    AZURE_VISION_ENDPOINT,
    AZURE_VISION_KEY,
    LLM_LITE_MODEL,
)
from utils import get_logger

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_REGISTRY: Dict[str, Dict[str, Any]] = {}   # name → {fn, mode, label, description}


def register_ocr_provider(
    name: str,
    *,
    mode: str,          # "vision" | "text"
    label: str,
    description: str,
):
    """
    Class-free decorator for registering an OCR provider.

    vision-mode fn signature : (image_bytes, media_type, **kw) -> str  [JSON]
    text-mode fn signature   : (image_bytes, media_type, **kw) -> str  [raw text]
    """
    def decorator(fn: Callable) -> Callable:
        _REGISTRY[name] = {
            "fn":          fn,
            "mode":        mode,
            "label":       label,
            "description": description,
        }
        log.debug("OCR provider registered: %s (%s mode)", name, mode)
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _ant_key() -> str:
    return os.environ.get("ANTHROPIC_API_KEY") or _CFG_ANT_KEY


def _oai_key() -> str:
    return os.environ.get("OPENAI_API_KEY") or _CFG_OAI_KEY


# The extraction prompt is shared — it tells the LLM exactly what fields to
# return whether it receives an image (vision mode) or a text transcription.
_FIELD_PROMPT = textwrap.dedent("""
You are a financial data extraction engine.

Extract the following fields from the provided financial data and return ONLY
a single valid JSON object.  Omit any field whose value is null or absent.
Do NOT include markdown fences, explanation, or extra keys.

SCALE: {scale_instruction}

JSON keys and their meaning:
  company_name            string  — company name if visible
  financial_year          string  — current year label (e.g. "FY2025")
  prior_financial_year    string  — prior year label  (e.g. "FY2024")

  revenue                 number  — current year, converted to raw INR
  ebitda                  number  — current year
  net_profit              number  — current year
  current_assets          number  — current year
  current_liabilities     number  — current year
  total_debt              number  — current year (borrowings)
  equity                  number  — current year (net worth / shareholders equity)
  cash_and_equivalents    number  — current year
  accounts_receivable     number  — current year (trade receivables / debtors)
  accounts_payable        number  — current year (trade payables / creditors)
  operating_cash_flow     number  — current year (cash from operations)
  interest_expense        number  — current year (finance costs)
  travel_cost             number  — current year, only if a travel / conveyance row exists

  revenue_py              number  — prior year equivalents for all above
  ebitda_py               number
  net_profit_py           number
  current_assets_py       number
  current_liabilities_py  number
  total_debt_py           number
  equity_py               number
  cash_and_equivalents_py number
  accounts_receivable_py  number
  accounts_payable_py     number
  operating_cash_flow_py  number
  interest_expense_py     number
  travel_cost_py          number

Rules:
- Negative values: brackets like (3.71) → -3.71
- Remove commas and currency symbols before converting to number
- If a value is unreadable or ambiguous use null (omit the key)
- Return ONLY the JSON object — no other text
""").strip()

_SCALE_INSTRUCTIONS = {
    10_000_000: "All values in the source are in ₹ Crores. Multiply each by 10,000,000 for raw INR.",
    100_000:    "All values in the source are in ₹ Lakhs. Multiply each by 100,000 for raw INR.",
    1_000_000:  "All values in the source are in ₹ Millions. Multiply each by 1,000,000 for raw INR.",
    1_000:      "All values in the source are in ₹ Thousands. Multiply each by 1,000 for raw INR.",
}
_SCALE_AUTO = (
    "Auto-detect scale from text clues like 'Figures in ₹ Crores', 'in Lakhs', etc. "
    "Apply: Crores×10,000,000 | Lakhs×100,000 | Millions×1,000,000 | Thousands×1,000 | "
    "No indicator→raw number."
)


def _scale_instruction(scale_override: Optional[float]) -> str:
    if scale_override and int(scale_override) in _SCALE_INSTRUCTIONS:
        return _SCALE_INSTRUCTIONS[int(scale_override)]
    return _SCALE_AUTO


def _build_extraction_prompt(scale_override: Optional[float]) -> str:
    return _FIELD_PROMPT.format(scale_instruction=_scale_instruction(scale_override))


# ---------------------------------------------------------------------------
# LLM-lite parse step (used by text-mode providers)
# ---------------------------------------------------------------------------

def _llm_lite_parse(
    raw_text: str,
    company_name: str,
    scale_override: Optional[float],
) -> Tuple[dict, List[str]]:
    """
    Use Claude Haiku (or GPT-4o-mini as fallback) to structure raw OCR text
    into the financial field dict.  This is a cheap, focused call — the heavy
    lifting (image → text) was done by the OCR provider.
    """
    warnings: List[str] = []
    prompt = (
        f"{_build_extraction_prompt(scale_override)}\n\n"
        f"--- RAW FINANCIAL DATA ---\n{raw_text}\n--- END ---"
    )

    raw_json: Optional[str] = None

    # Try Anthropic Haiku first
    if _ant_key():
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=_ant_key())
            msg = client.messages.create(
                model=LLM_LITE_MODEL,
                max_tokens=2048,      # ← was 1024
                messages=[{"role": "user", "content": prompt}],
            )
            raw_json = msg.content[0].text.strip()
            log.info("LLM-lite parse: used Anthropic %s", LLM_LITE_MODEL)
        except Exception as exc:
            log.warning("Anthropic lite parse failed: %s", exc)

    # Fallback: OpenAI gpt-4o-mini
    if raw_json is None and _oai_key():
        try:
            from openai import OpenAI
            client = OpenAI(api_key=_oai_key())
            resp = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=2048,      # ← was 1024
                response_format={"type": "json_object"},
            )
            raw_json = resp.choices[0].message.content
            log.info("LLM-lite parse: used OpenAI gpt-4o-mini (fallback)")
        except Exception as exc:
            log.warning("OpenAI lite parse fallback failed: %s", exc)

    if raw_json is None:
        warnings.append(
            "⚠️ No LLM key available to structure the OCR text. "
            "Add ANTHROPIC_API_KEY or OPENAI_API_KEY to enable structured extraction."
        )
        return {}, warnings

    # Parse the JSON
    try:
        import re
        text = re.sub(r"```(?:json)?", "", raw_json).strip("` \n")
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        warnings.append(f"⚠️ JSON parse error in LLM-lite response: {exc}")
        return {}, warnings

    if company_name and not parsed.get("company_name"):
        parsed["company_name"] = company_name

    # Filter out None / sentinel values
    result = {k: v for k, v in parsed.items() if v is not None}

    _core = {"revenue", "net_profit", "equity", "total_debt", "operating_cash_flow"}
    missing = _core - set(result)
    if missing:
        warnings.append(
            f"⚠️ Could not extract: **{', '.join(sorted(missing))}**. "
            "Fill these in the preview before running the evaluation."
        )

    extracted_n = sum(1 for k in result if not k.startswith("financial") and k != "company_name")
    warnings.append(
        f"✅ **{extracted_n} fields** structured by LLM-lite from OCR text. "
        "Verify figures match the original document."
    )
    return result, warnings


# ---------------------------------------------------------------------------
# Vision-mode helpers (shared extraction prompt for image)
# ---------------------------------------------------------------------------

def _vision_prompt(scale_override: Optional[float]) -> str:
    """Full prompt for vision providers — references the image directly."""
    return textwrap.dedent(f"""
    You are a financial data extraction engine.

    Read the uploaded image of a financial statement table (P&L, Balance Sheet,
    or Cash Flow) and extract every figure.  The table typically has:
      Column 1 : Metric label
      Column 2 : Current Year  (higher year number)
      Column 3 : Prior Year    (lower year number)

    {_build_extraction_prompt(scale_override)}

    OCR correction: fix common misreads — l→1, O→0, I→1, S→5, B→8.
    Preserve negative values: (3.71) → -3.71.
    If a value is genuinely unreadable, omit that key.
    Return ONLY the JSON object.
    """).strip()


# ---------------------------------------------------------------------------
# Built-in providers
# ---------------------------------------------------------------------------

@register_ocr_provider(
    "claude",
    mode        = "vision",
    label       = "Claude Haiku ✦ (Recommended)",
    description = "Anthropic's fast vision model. Needs ANTHROPIC_API_KEY.",
)
def _provider_claude(
    image_bytes: bytes,
    media_type: str,
    company_name: str = "",
    scale_override: Optional[float] = None,
    **_,
) -> str:
    """Single-shot: image → structured JSON via Claude Haiku."""
    import anthropic
    key = _ant_key()
    if not key:
        raise ValueError("ANTHROPIC_API_KEY is not set — cannot run Claude vision OCR")
    client = anthropic.Anthropic(api_key=key)
    b64 = base64.standard_b64encode(image_bytes).decode()
    log.info(
        "Claude vision OCR: model=%s image_size=%d bytes media_type=%s",
        LLM_LITE_MODEL, len(image_bytes), media_type,
    )
    msg = client.messages.create(
        model=LLM_LITE_MODEL,
        max_tokens=4096,          # ← was 1024; full financial JSON can be 2-3k tokens
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": media_type, "data": b64}},
                {"type": "text", "text": _vision_prompt(scale_override)},
            ],
        }],
    )
    raw = msg.content[0].text.strip()
    log.info(
        "Claude vision OCR complete: stop_reason=%s output_tokens=%d",
        msg.stop_reason, msg.usage.output_tokens,
    )
    return raw


@register_ocr_provider(
    "openai",
    mode        = "vision",
    label       = "GPT-4o mini",
    description = "OpenAI's cost-efficient vision model. Needs OPENAI_API_KEY.",
)
def _provider_openai(
    image_bytes: bytes,
    media_type: str,
    company_name: str = "",
    scale_override: Optional[float] = None,
    **_,
) -> str:
    """Single-shot: image → structured JSON via GPT-4o-mini."""
    from openai import OpenAI
    client = OpenAI(api_key=_oai_key())
    b64 = base64.standard_b64encode(image_bytes).decode()
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url",
                 "image_url": {"url": f"data:{media_type};base64,{b64}"}},
                {"type": "text", "text": _vision_prompt(scale_override)},
            ],
        }],
        max_tokens=4096,          # ← was 1024; full financial JSON can be 2-3k tokens
        response_format={"type": "json_object"},
    )
    return resp.choices[0].message.content


@register_ocr_provider(
    "tesseract",
    mode        = "text",
    label       = "Tesseract OCR (Local)",
    description = "Free local OCR. Requires: pip install pytesseract pillow + Tesseract binary.",
)
def _provider_tesseract(
    image_bytes: bytes,
    media_type: str,
    **_,
) -> str:
    """Extract raw text from image using local Tesseract."""
    import pytesseract
    from PIL import Image
    img = Image.open(io.BytesIO(image_bytes))
    # PSM 6: treat as single uniform block of text — good for tables
    raw = pytesseract.image_to_string(img, config="--psm 6")
    log.info("Tesseract extracted %d chars", len(raw))
    return raw


@register_ocr_provider(
    "google_vision",
    mode        = "text",
    label       = "Google Cloud Vision",
    description = "Google's document OCR API. Needs GOOGLE_VISION_API_KEY.",
)
def _provider_google_vision(
    image_bytes: bytes,
    media_type: str,
    **_,
) -> str:
    """Extract raw text using Google Cloud Vision document_text_detection."""
    import requests as _req
    key = GOOGLE_VISION_API_KEY or os.environ.get("GOOGLE_VISION_API_KEY", "")
    if not key:
        raise ValueError("GOOGLE_VISION_API_KEY is not set")
    b64 = base64.standard_b64encode(image_bytes).decode()
    payload = {
        "requests": [{
            "image":    {"content": b64},
            "features": [{"type": "DOCUMENT_TEXT_DETECTION"}],
        }]
    }
    resp = _req.post(
        f"https://vision.googleapis.com/v1/images:annotate?key={key}",
        json=payload, timeout=15,
    )
    resp.raise_for_status()
    data = resp.json()
    text = (data.get("responses", [{}])[0]
                .get("fullTextAnnotation", {})
                .get("text", ""))
    log.info("Google Vision extracted %d chars", len(text))
    return text


@register_ocr_provider(
    "azure_vision",
    mode        = "text",
    label       = "Azure Computer Vision",
    description = "Azure Read API. Needs AZURE_VISION_ENDPOINT + AZURE_VISION_KEY.",
)
def _provider_azure_vision(
    image_bytes: bytes,
    media_type: str,
    **_,
) -> str:
    """Extract raw text using Azure Computer Vision Read API (REST, no SDK needed)."""
    import time as _time
    import requests as _req

    endpoint = AZURE_VISION_ENDPOINT or os.environ.get("AZURE_VISION_ENDPOINT", "")
    key      = AZURE_VISION_KEY      or os.environ.get("AZURE_VISION_KEY", "")
    if not endpoint or not key:
        raise ValueError("AZURE_VISION_ENDPOINT and AZURE_VISION_KEY must both be set")

    base = endpoint.rstrip("/")
    submit_url = f"{base}/vision/v3.2/read/analyze"
    headers = {
        "Ocp-Apim-Subscription-Key": key,
        "Content-Type": media_type,
    }
    # Submit the image for async analysis
    resp = _req.post(submit_url, headers=headers, data=image_bytes, timeout=15)
    resp.raise_for_status()
    operation_url = resp.headers.get("Operation-Location", "")
    if not operation_url:
        raise RuntimeError("Azure: no Operation-Location header in response")

    # Poll for result (max 20s)
    poll_headers = {"Ocp-Apim-Subscription-Key": key}
    for _ in range(10):
        _time.sleep(2)
        poll = _req.get(operation_url, headers=poll_headers, timeout=10)
        poll.raise_for_status()
        result = poll.json()
        status = result.get("status", "")
        if status == "succeeded":
            lines = []
            for page in result.get("analyzeResult", {}).get("readResults", []):
                for line in page.get("lines", []):
                    lines.append(line.get("text", ""))
            text = "\n".join(lines)
            log.info("Azure Vision extracted %d chars", len(text))
            return text
        if status == "failed":
            raise RuntimeError(f"Azure Read API analysis failed: {result}")

    raise TimeoutError("Azure Read API did not complete within 20 seconds")


# ---------------------------------------------------------------------------
# Availability checking
# ---------------------------------------------------------------------------

def check_provider_availability(name: str) -> Tuple[bool, str]:
    """
    Return (available: bool, reason: str).
    Reason is empty string when available.
    """
    if name not in _REGISTRY:
        return False, f"Provider '{name}' is not registered"

    checks = {
        "claude":        lambda: (bool(_ant_key()),       "ANTHROPIC_API_KEY not set"),
        "openai":        lambda: (bool(_oai_key()),       "OPENAI_API_KEY not set"),
        "tesseract":     _check_tesseract,
        "google_vision": lambda: (bool(GOOGLE_VISION_API_KEY or os.environ.get("GOOGLE_VISION_API_KEY")),
                                  "GOOGLE_VISION_API_KEY not set"),
        "azure_vision":  lambda: (
            bool((AZURE_VISION_ENDPOINT or os.environ.get("AZURE_VISION_ENDPOINT"))
                 and (AZURE_VISION_KEY or os.environ.get("AZURE_VISION_KEY"))),
            "AZURE_VISION_ENDPOINT and/or AZURE_VISION_KEY not set"),
    }
    check_fn = checks.get(name)
    if check_fn is None:
        # Custom / external provider — assume available
        return True, ""
    available, reason = check_fn()
    return available, ("" if available else reason)


def _check_tesseract() -> Tuple[bool, str]:
    try:
        import pytesseract  # noqa: F401
        from PIL import Image  # noqa: F401
        pytesseract.get_tesseract_version()
        return True, ""
    except ImportError:
        return False, "pytesseract / Pillow not installed: pip install pytesseract pillow"
    except Exception as exc:
        return False, f"Tesseract binary not found: {exc}"


def list_providers() -> List[Dict[str, Any]]:
    """
    Return all registered providers, each annotated with availability status.

    Each entry:
      name, label, description, mode, available (bool), unavailable_reason (str)
    """
    out = []
    for name, meta in _REGISTRY.items():
        avail, reason = check_provider_availability(name)
        out.append({
            "name":               name,
            "label":              meta["label"],
            "description":        meta["description"],
            "mode":               meta["mode"],
            "available":          avail,
            "unavailable_reason": reason,
        })
    return out


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def extract_financials(
    image_bytes: bytes,
    media_type: str,
    provider: str,
    company_name: str = "",
    scale_override: Optional[float] = None,
) -> Tuple[dict, List[str]]:
    """
    Extract financial fields from an image using the named OCR provider.

    Returns (financials_dict, warnings_list).
    financials_dict keys match those expected by the scoring / credit engines.
    """
    if provider not in _REGISTRY:
        available = [p["name"] for p in list_providers() if p["available"]]
        return {}, [
            f"OCR provider '{provider}' is not registered. "
            f"Available providers: {', '.join(available) or 'none'}."
        ]

    avail, reason = check_provider_availability(provider)
    if not avail:
        return {}, [f"OCR provider '{provider}' is not available: {reason}"]

    meta = _REGISTRY[provider]
    mode = meta["mode"]
    fn   = meta["fn"]

    log.info("OCR extraction: provider=%s mode=%s company=%s", provider, mode, company_name)

    try:
        raw = fn(
            image_bytes,
            media_type,
            company_name   = company_name,
            scale_override = scale_override,
        )
    except Exception as exc:
        log.error("OCR provider '%s' failed: %s", provider, exc, exc_info=True)
        return {}, [f"OCR provider '{provider}' error: {exc}"]

    warnings: List[str] = [
        f"📷 OCR provider: **{meta['label']}** ({mode} mode)"
    ]

    # ── Vision mode: raw is already a JSON string ────────────────────────────
    if mode == "vision":
        try:
            import re
            text = re.sub(r"```(?:json)?", "", raw).strip("` \n")

            # Fix literal newlines inside JSON strings (LLM quirk)
            from ai_engine import _sanitise_json_strings
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                parsed = json.loads(_sanitise_json_strings(text))

        except json.JSONDecodeError as exc:
            warnings.append(f"⚠️ JSON parse error from {meta['label']}: {exc}")
            return {}, warnings

        if company_name and not parsed.get("company_name"):
            parsed["company_name"] = company_name

        result = {k: v for k, v in parsed.items() if v is not None}
        _core  = {"revenue", "net_profit", "equity", "total_debt", "operating_cash_flow"}
        missing = _core - set(result)
        if missing:
            warnings.append(
                f"⚠️ Could not extract: **{', '.join(sorted(missing))}**. "
                "Fill these in the preview before running the evaluation."
            )
        n = sum(1 for k in result if k not in ("company_name", "financial_year",
                                                "prior_financial_year"))
        warnings.append(
            f"✅ **{n} financial fields** extracted via {meta['label']}. "
            "Verify figures match the original document."
        )
        log.info("Vision extraction complete: %d fields", n)
        return result, warnings

    # ── Text mode: raw is OCR text → run LLM-lite parse ─────────────────────
    if not raw.strip():
        warnings.append("⚠️ OCR provider returned empty text — check the image quality.")
        return {}, warnings

    log.info("Text OCR returned %d chars — running LLM-lite parse", len(raw))
    parsed, parse_warnings = _llm_lite_parse(raw, company_name, scale_override)
    warnings.extend(parse_warnings)
    return parsed, warnings
