"""
SQLite persistence layer.
Schema is auto-created on first run; no migrations needed for v1.
"""

from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

from config import DB_PATH
from utils import get_logger, timestamp_now

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_DDL = """
CREATE TABLE IF NOT EXISTS evaluations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    company_name    TEXT    NOT NULL,
    source_file     TEXT,
    -- raw extracted financials (JSON)
    financials_json TEXT,
    parse_confidence REAL,
    -- payment behavior (JSON)
    payment_json    TEXT,
    -- scoring results (JSON)
    scoring_json    TEXT,
    -- external signals (JSON)
    external_json   TEXT,
    -- credit decision (JSON)
    decision_json   TEXT,
    -- composite score
    composite_score REAL,
    risk_category   TEXT,
    credit_limit    REAL,
    payment_terms   INTEGER,
    -- AI narrative
    ai_narrative    TEXT,
    ai_confidence   REAL,
    -- status
    status          TEXT    DEFAULT 'complete'
);

CREATE TABLE IF NOT EXISTS payment_history (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    company_name    TEXT    NOT NULL,
    invoice_date    TEXT,
    due_date        TEXT,
    paid_date       TEXT,
    amount          REAL,
    days_late       INTEGER,
    status          TEXT    -- 'ontime' | 'late_30' | 'late_60' | 'late_90' | 'default'
);

CREATE INDEX IF NOT EXISTS idx_eval_company ON evaluations(company_name);
CREATE INDEX IF NOT EXISTS idx_eval_created ON evaluations(created_at);
CREATE INDEX IF NOT EXISTS idx_pay_company  ON payment_history(company_name);
"""


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def _conn():
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


def init_db() -> None:
    """Create tables if they don't exist."""
    with _conn() as con:
        con.executescript(_DDL)
    log.info("Database initialised at %s", DB_PATH)


# ---------------------------------------------------------------------------
# Evaluation CRUD
# ---------------------------------------------------------------------------

def save_evaluation(
    company_name: str,
    financials: dict,
    payment_data: dict,
    scoring_result: dict,
    external_result: dict,
    decision: dict,
    source_file: str = "",
    parse_confidence: float = 1.0,
    ai_narrative: str = "",
    ai_confidence: float = 0.0,
) -> int:
    """Persist a full evaluation; returns the new row id."""
    row = {
        "created_at":       timestamp_now(),
        "company_name":     company_name,
        "source_file":      source_file,
        "financials_json":  json.dumps(financials),
        "parse_confidence": parse_confidence,
        "payment_json":     json.dumps(payment_data),
        "scoring_json":     json.dumps(scoring_result),
        "external_json":    json.dumps(external_result),
        "decision_json":    json.dumps(decision),
        "composite_score":  decision.get("composite_score", 0.0),
        "risk_category":    decision.get("risk_category", "Unknown"),
        "credit_limit":     decision.get("credit_limit", 0.0),
        "payment_terms":    decision.get("payment_terms", 30),
        "ai_narrative":     ai_narrative,
        "ai_confidence":    ai_confidence,
    }
    cols = ", ".join(row.keys())
    placeholders = ", ".join("?" for _ in row)
    sql = f"INSERT INTO evaluations ({cols}) VALUES ({placeholders})"
    with _conn() as con:
        cur = con.execute(sql, list(row.values()))
        row_id = cur.lastrowid
    log.info("Saved evaluation id=%d for %s", row_id, company_name)
    return row_id


def list_evaluations(limit: int = 50) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT id, created_at, company_name, composite_score, risk_category, "
            "credit_limit, payment_terms, source_file FROM evaluations ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]


def search_evaluations(
    query: str = "",
    risk_filter: str = "All",
    limit: int = 200,
) -> List[Dict[str, Any]]:
    """
    Search evaluations by company name (case-insensitive substring match)
    with optional risk category filter.
    """
    base_sql = (
        "SELECT id, created_at, company_name, composite_score, risk_category, "
        "credit_limit, payment_terms, source_file FROM evaluations"
    )
    conditions: List[str] = []
    params: List[Any] = []

    if query.strip():
        conditions.append("LOWER(company_name) LIKE LOWER(?)")
        params.append(f"%{query.strip()}%")

    if risk_filter and risk_filter != "All":
        conditions.append("risk_category = ?")
        params.append(risk_filter)

    if conditions:
        base_sql += " WHERE " + " AND ".join(conditions)

    base_sql += " ORDER BY created_at DESC LIMIT ?"
    params.append(limit)

    with _conn() as con:
        rows = con.execute(base_sql, params).fetchall()
    return [dict(r) for r in rows]


def get_evaluation(eval_id: int) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM evaluations WHERE id = ?", (eval_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    for key in ("financials_json", "payment_json", "scoring_json", "external_json", "decision_json"):
        if d.get(key):
            d[key.replace("_json", "")] = json.loads(d[key])
    return d


def get_latest_evaluation(company_name: str) -> Optional[Dict[str, Any]]:
    with _conn() as con:
        row = con.execute(
            "SELECT id FROM evaluations WHERE company_name = ? ORDER BY created_at DESC LIMIT 1",
            (company_name,),
        ).fetchone()
    if row is None:
        return None
    return get_evaluation(row["id"])


# ---------------------------------------------------------------------------
# Payment history CRUD
# ---------------------------------------------------------------------------

def save_payment_records(records: List[Dict[str, Any]]) -> None:
    if not records:
        return
    cols = ["company_name", "invoice_date", "due_date", "paid_date", "amount", "days_late", "status"]
    sql = f"INSERT INTO payment_history ({', '.join(cols)}) VALUES ({', '.join('?' for _ in cols)})"
    rows = [[r.get(c) for c in cols] for r in records]
    with _conn() as con:
        con.executemany(sql, rows)
    log.info("Saved %d payment records", len(rows))


def get_payment_history(company_name: str) -> List[Dict[str, Any]]:
    with _conn() as con:
        rows = con.execute(
            "SELECT * FROM payment_history WHERE company_name = ? ORDER BY due_date DESC",
            (company_name,),
        ).fetchall()
    return [dict(r) for r in rows]
