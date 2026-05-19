"""
Run this once to understand the exact layout of both PDFs.
Output is written to logs/pdf_analysis.txt
"""
import sys, pdfplumber
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

PDFS = {
    "Prestige Estates": r"C:\Users\Inno\Downloads\Financials\Prestige estates fy25.pdf",
    "PPG Asian Paints": r"C:\Users\Inno\Downloads\Financials\PPG Asian Paints FY25.pdf",
}

FINANCIAL_KEYWORDS = [
    "revenue from operations", "profit for the year", "profit after tax",
    "total equity", "total assets", "net cash", "finance cost",
    "total income", "borrowings", "rs. in", "in crore", "in lakh",
    "in million", "balance sheet", "statement of profit", "cash flow",
    "ebidta", "ebitda", "pbdit",
]

for company, path in PDFS.items():
    print(f"\n{'='*70}")
    print(f"COMPANY: {company}")
    print(f"FILE:    {path}")
    print(f"{'='*70}")
    try:
        with pdfplumber.open(path) as pdf:
            print(f"Total pages: {len(pdf.pages)}\n")
            hit_pages = []
            for i, page in enumerate(pdf.pages):
                text = (page.extract_text() or "").lower()
                if any(k in text for k in FINANCIAL_KEYWORDS):
                    hit_pages.append(i)

            # Show first 6 matching pages
            for pg_idx in hit_pages[:6]:
                page = pdf.pages[pg_idx]
                text = page.extract_text() or ""
                print(f"\n--- PAGE {pg_idx+1} (text extract) ---")
                print(text[:2500])

                # Also show table data if present
                tables = page.extract_tables()
                if tables:
                    print(f"\n  [TABLE DATA — {len(tables)} table(s)]")
                    for t_idx, table in enumerate(tables[:2]):
                        print(f"  Table {t_idx+1}:")
                        for row in table[:20]:
                            if row and any(c for c in row if c):
                                print("   ", row)
    except Exception as e:
        print(f"ERROR: {e}")
