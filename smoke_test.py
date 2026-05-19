"""Quick end-to-end smoke test — run with: python smoke_test.py"""
import load_env
from dataclasses import asdict

from scoring import run_scoring, scoring_result_to_dict
from ai_engine import run_ai_analysis
from external_data import fetch_external_risk
from credit_engine import compute_credit_decision
from database import init_db, save_evaluation, list_evaluations

fin = {
    "company_name":        "Apex Logistics International",
    "revenue":             12_000_000,
    "ebitda":              2_400_000,
    "net_profit":          1_400_000,
    "current_assets":      4_500_000,
    "current_liabilities": 1_800_000,
    "total_debt":          3_000_000,
    "equity":              6_000_000,
    "cash_and_equivalents":  900_000,
    "operating_cash_flow": 1_800_000,
    "interest_expense":      240_000,
}

print("1/5  Fetching external risk (mock)...")
ext = fetch_external_risk("Apex Logistics", use_mock=True)
print(f"     Sentiment: {ext['sentiment']}, flags: {len(ext['risk_flags'])}")

print("2/5  Running scoring model...")
scoring = run_scoring(fin, [], ext)
sd = scoring_result_to_dict(scoring)
print(f"     Composite: {scoring.composite_score:.1f}/100  Band: {scoring.risk_category}")

print("3/5  Running AI analysis (mock)...")
ai = run_ai_analysis(fin, [asdict(r) for r in scoring.ratios], sd, ext, {})
print(f"     Recommendation: {ai.recommendation}  Confidence: {ai.confidence_level}")

print("4/5  Computing credit limit...")
decision = compute_credit_decision(fin, scoring, ext, ai.recommendation)
print(f"     Limit: ${decision.credit_limit:,.0f}  Terms: Net-{decision.payment_terms_days}")

print("5/5  Saving to SQLite and reading back...")
init_db()
eid = save_evaluation(
    company_name="Apex Logistics International",
    financials=fin, payment_data={}, scoring_result=sd,
    external_result=ext,
    decision={
        "composite_score": scoring.composite_score,
        "risk_category":   scoring.risk_category,
        "credit_limit":    decision.credit_limit,
        "payment_terms":   decision.payment_terms_days,
    },
    ai_narrative=ai.risk_narrative,
    ai_confidence=ai.confidence_score,
)
rows = list_evaluations(limit=5)
print(f"     Saved eval_id={eid}, total rows={len(rows)}")

print()
print("=" * 50)
print("  END-TO-END SMOKE TEST PASSED")
print("=" * 50)
print()
print("Summary:")
print(f"  Company   : {fin['company_name']}")
print(f"  Score     : {scoring.composite_score:.1f}/100")
print(f"  Category  : {scoring.risk_category}  [{scoring.risk_emoji}]".encode('ascii', errors='replace').decode('ascii'))
print(f"  AI Dec.   : {ai.recommendation}")
print(f"  Limit     : ${decision.credit_limit:,.0f}")
print(f"  Terms     : Net-{decision.payment_terms_days}")
print(f"  Red flags : {len(scoring.red_flags)}")
