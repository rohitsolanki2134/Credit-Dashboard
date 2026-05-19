"""
Test the parser against the two real PDFs.
Run: python test_pdfs.py
"""
import sys, json
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from parser import parse_pdf
from utils import fmt_currency

PDFS = {
    "PPG Asian Paints FY25":
        r"C:\Users\Inno\Downloads\Financials\PPG Asian Paints FY25.pdf",
    "Prestige Estates FY25":
        r"C:\Users\Inno\Downloads\Financials\Prestige estates fy25.pdf",
}

# Expected values (from manual reading of the PDFs) — in absolute rupees
EXPECTED = {
    "PPG Asian Paints FY25": {
        # Rs. in million × 1,000,000
        "revenue":              21_423.23e6,
        "net_profit":            2_814.14e6,
        "total_assets":         17_139.74e6,
        "equity":               12_159.98e6,
        "current_assets":       12_327.91e6,
        "current_liabilities":   4_559.85e6,
        "interest_expense":         22.61e6,
        "operating_cash_flow":   2_069.06e6,
    },
    "Prestige Estates FY25": {
        # Rs. in Mn × 1,000,000 — consolidated figures
        "revenue":              73_494e6,
        "net_profit":            6_169e6,
        "ebitda":               29_449e6,
    },
}

FIELD_ORDER = [
    "company_name", "revenue", "ebitda", "net_profit",
    "total_debt", "equity", "current_assets", "current_liabilities",
    "cash_and_equivalents", "operating_cash_flow", "interest_expense",
    "total_assets",
]

for name, path in PDFS.items():
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"{'='*65}")
    try:
        data, conf, warnings = parse_pdf(open(path, "rb").read(), company_name="")
        print(f"  Company      : {data.get('company_name', 'N/A')}")
        print(f"  Confidence   : {conf:.0%}")
        print()
        print(f"  {'Field':<28}  {'Extracted':>18}  {'Expected':>18}  {'Match'}")
        print(f"  {'-'*28}  {'-'*18}  {'-'*18}  {'-'*5}")
        exp = EXPECTED.get(name, {})
        for field in FIELD_ORDER:
            val = data.get(field)
            exp_val = exp.get(field)
            if val is None and exp_val is None:
                continue
            if field == "company_name":
                print(f"  {'company_name':<28}  {str(val)[:35]:<35}")
                continue
            try:
                extracted_s = fmt_currency(float(val)) if val is not None else "—"
            except (TypeError, ValueError):
                extracted_s = str(val)[:18]
            expected_s  = fmt_currency(exp_val) if exp_val else "—"
            if val is not None and exp_val is not None:
                try:
                    ratio = float(val) / exp_val
                    ok = "OK" if 0.80 <= ratio <= 1.20 else f"DIFF({ratio:.2f}x)"
                except Exception:
                    ok = "?"
            elif val is not None:
                ok = "EXTRA"
            else:
                ok = "MISS"
            print(f"  {field:<28}  {extracted_s:>18}  {expected_s:>18}  {ok}")
        print()
        if warnings:
            print("  Warnings:")
            for w in warnings:
                print(f"    - {w}")
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback; traceback.print_exc()
