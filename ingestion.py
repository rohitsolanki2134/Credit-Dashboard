"""
Ingestion layer: accepts raw uploads (PDF / CSV / Excel / dict),
validates them, and routes to the appropriate parser.
"""

from __future__ import annotations
import io
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from utils import get_logger, validate_financials

log = get_logger(__name__)

# Supported MIME types / extensions
SUPPORTED_EXTENSIONS = {".pdf", ".csv", ".xlsx", ".xls", ".png", ".jpg", ".jpeg", ".webp"}

_IMAGE_MEDIA_TYPES = {
    ".png":  "image/png",
    ".jpg":  "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ingest(
    source: Union[bytes, str, Path, dict],
    filename: str = "",
    company_name: str = "",
    scale_override: Optional[float] = None,
    ocr_provider: Optional[str] = None,
) -> Tuple[dict, float, list[str]]:
    """
    Accept any supported input and return:
      (financial_data_dict, parse_confidence_0_to_1, warnings_list)

    source can be:
      - bytes  : raw file bytes (from Streamlit uploader)
      - str/Path: filesystem path to a file
      - dict   : already-structured financial data (pass-through)
    """
    if isinstance(source, dict):
        return _passthrough(source)

    if isinstance(source, (str, Path)):
        source = Path(source)
        filename = filename or source.name
        with open(source, "rb") as fh:
            raw_bytes = fh.read()
    else:
        raw_bytes = source  # bytes from Streamlit

    ext = Path(filename).suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type '{ext}'. Supported: {SUPPORTED_EXTENSIONS}")

    log.info("Ingesting file: %s (%d bytes)", filename, len(raw_bytes))

    if ext in _IMAGE_MEDIA_TYPES:
        from ocr_engine import extract_financials as _ocr_extract, list_providers
        from config import OCR_PROVIDER

        media_type = _IMAGE_MEDIA_TYPES[ext]

        # Resolve which provider to use
        provider = ocr_provider or OCR_PROVIDER

        # Auto-fallback: if the requested provider is unavailable, pick the
        # first available one (prefer vision-mode providers)
        available = [p for p in list_providers() if p["available"]]
        available_names = [p["name"] for p in available]
        if provider not in available_names:
            vision_first = sorted(available, key=lambda p: 0 if p["mode"] == "vision" else 1)
            if vision_first:
                fallback = vision_first[0]["name"]
                log.warning(
                    "Requested OCR provider '%s' unavailable — falling back to '%s'",
                    provider, fallback,
                )
                provider = fallback
            else:
                # Absolute last resort: legacy local OCR parser
                log.warning("No OCR provider available — using legacy local parser")
                from image_parser import parse_image
                data, confidence, warnings = parse_image(raw_bytes, scale_override=scale_override)
                if company_name and not data.get("company_name"):
                    data["company_name"] = company_name
                return data, confidence, warnings

        data, warnings = _ocr_extract(
            raw_bytes, media_type,
            provider       = provider,
            company_name   = company_name,
            scale_override = scale_override,
        )
        confidence = 0.85 if data.get("revenue") else 0.35
    elif ext == ".pdf":
        from parser import parse_pdf
        data, confidence, warnings = parse_pdf(raw_bytes, company_name=company_name)
    elif ext in (".csv",):
        from parser import parse_csv
        data, confidence, warnings = parse_csv(raw_bytes)
    else:  # .xlsx / .xls
        from parser import parse_excel
        data, confidence, warnings = parse_excel(raw_bytes)

    # Merge in company name if parser didn't find one
    if company_name and not data.get("company_name"):
        data["company_name"] = company_name

    is_valid, validation_warnings = validate_financials(data)
    warnings.extend(validation_warnings)

    log.info(
        "Ingestion complete | confidence=%.2f | warnings=%d | valid=%s",
        confidence, len(warnings), is_valid,
    )
    return data, confidence, warnings


def ingest_payment_csv(source: Union[bytes, str, Path], filename: str = None) -> list[dict]:
    """
    Parse a payment-history file (CSV or Excel).

    Accepted columns:
      Required : invoice_date, due_date, paid_date, amount | amount_inr
      Optional : invoice_number, description, currency, company_name
    """
    buf = io.BytesIO(source) if isinstance(source, (bytes, bytearray)) else source

    # Detect Excel by filename extension or by trying openpyxl
    is_excel = False
    if filename and filename.lower().endswith((".xlsx", ".xls")):
        is_excel = True

    date_cols = ["invoice_date", "due_date", "paid_date"]

    if is_excel:
        df = pd.read_excel(buf, sheet_name="Payment History" if filename else 0)
    else:
        try:
            df = pd.read_csv(buf)
        except Exception:
            buf.seek(0)
            df = pd.read_excel(buf)

    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    # Normalise amount column: accept "amount_inr" as alias for "amount"
    if "amount" not in df.columns and "amount_inr" in df.columns:
        df = df.rename(columns={"amount_inr": "amount"})

    required = {"invoice_date", "due_date", "paid_date", "amount"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Payment file missing columns: {missing}")

    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    records = []
    for _, row in df.iterrows():
        paid_raw = row.get("paid_date")
        due_raw  = row.get("due_date")
        try:
            paid = pd.to_datetime(paid_raw)
            due  = pd.to_datetime(due_raw)
            if pd.isna(paid):
                # Unpaid invoice — outstanding/overdue
                days_late = None
                status    = "unpaid"
            else:
                days_late = max(0, (paid - due).days)
                if days_late == 0:
                    status = "ontime"
                elif days_late <= 30:
                    status = "late_30"
                elif days_late <= 60:
                    status = "late_60"
                elif days_late <= 90:
                    status = "late_90"
                else:
                    status = "default"
        except Exception:
            days_late = None
            status = "unknown"

        records.append({
            "company_name":   str(row.get("company_name", "")),
            "invoice_number": str(row.get("invoice_number", "")),
            "description":    str(row.get("description", "")),
            "invoice_date":   str(row.get("invoice_date", "")),
            "due_date":       str(row.get("due_date", "")),
            "paid_date":      str(row.get("paid_date", "")),
            "amount":         float(row.get("amount", 0)),
            "days_late":      days_late,
            "status":         status,
        })

    log.info("Parsed %d payment records from %s", len(records), filename or "bytes")
    return records


# ---------------------------------------------------------------------------
# Passthrough for already-structured dicts
# ---------------------------------------------------------------------------

def _passthrough(data: dict) -> Tuple[dict, float, list[str]]:
    _, warnings = validate_financials(data)
    confidence = 1.0 if not warnings else 0.85
    return data, confidence, warnings
