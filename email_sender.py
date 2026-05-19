"""
email_sender.py
───────────────
CFO-style credit evaluation email builder and SMTP dispatcher.

Usage
-----
    from email_sender import build_cfo_email, send_email, send_result
"""

from __future__ import annotations

import smtplib
import ssl
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Tuple

from utils import get_logger, fmt_currency

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# HTML email template builder
# ---------------------------------------------------------------------------

def _fmt_inr(value) -> str:
    """
    Format a raw-INR value for display in the email.
    Uses the same logic as the dashboard (fmt_currency from utils).
    Returns '—' for None / missing values.
    """
    if value is None:
        return "—"
    try:
        return fmt_currency(float(value))
    except Exception:
        return str(value)


def _signal_badge(signal: str) -> str:
    styles = {
        "good":     ("background:#D1FAE5;color:#065F46", "GOOD"),
        "ok":       ("background:#DBEAFE;color:#1E40AF", "OK"),
        "warning":  ("background:#FEF3C7;color:#92400E", "WARNING"),
        "critical": ("background:#FEE2E2;color:#991B1B", "CRITICAL"),
    }
    style, label = styles.get(signal, ("background:#F1F5F9;color:#475569", signal.upper()))
    return (
        f'<span style="{style};padding:2px 10px;border-radius:20px;'
        f'font-size:11px;font-weight:700;white-space:nowrap">{label}</span>'
    )


def build_cfo_email(result: dict) -> Tuple[str, str]:
    """
    Build a CFO-style credit evaluation email.

    Returns
    -------
    subject : str
    html_body : str
    """
    scoring   = result["scoring"]
    decision  = result["decision"]
    ai        = result["ai"]
    financials = result["financials"]
    ext        = result.get("external", {})
    company    = result.get("company_name", "Unknown Company")
    eval_date  = date.today().strftime("%d %B %Y")

    # ── Decision colours ────────────────────────────────────────────────────
    approved       = decision.approved
    decision_label = "APPROVED" if approved else "DECLINED"
    decision_color = "#16A34A" if approved else "#DC2626"
    decision_bg    = "#D1FAE5" if approved else "#FEE2E2"

    ai_rec   = ai.recommendation          # APPROVE / CONDITIONAL / REJECT
    ai_color = {"APPROVE": "#16A34A", "CONDITIONAL": "#D97706", "REJECT": "#DC2626"}.get(ai_rec, "#475569")

    score   = scoring.composite_score
    band    = "Good" if score >= 70 else "Average" if score >= 45 else "Risky"
    band_bg = "#D1FAE5" if band == "Good" else "#FEF3C7" if band == "Average" else "#FEE2E2"
    band_fg = "#065F46" if band == "Good" else "#92400E" if band == "Average" else "#991B1B"

    subject = f"Credit Evaluation Report — {company} | {decision_label} | Score {score:.0f}/100"

    # ── Sub-score rows ───────────────────────────────────────────────────────
    sub_rows = ""
    for ss in scoring.sub_scores:
        pct       = ss.score
        bar_color = "#16A34A" if pct >= 70 else "#D97706" if pct >= 45 else "#DC2626"
        sub_rows += f"""
        <tr>
          <td style="padding:6px 12px;border-bottom:1px solid #F1F5F9;color:#374151;font-size:13px">{ss.name}</td>
          <td style="padding:6px 12px;border-bottom:1px solid #F1F5F9;color:#6B7280;font-size:12px;text-align:center">{ss.weight*100:.0f}%</td>
          <td style="padding:6px 12px;border-bottom:1px solid #F1F5F9;min-width:160px">
            <div style="display:flex;align-items:center;gap:8px">
              <div style="flex:1;background:#E5E7EB;border-radius:4px;height:8px">
                <div style="background:{bar_color};width:{pct:.0f}%;height:8px;border-radius:4px"></div>
              </div>
              <span style="font-size:12px;font-weight:700;color:{bar_color};min-width:36px">{pct:.0f}/100</span>
            </div>
          </td>
        </tr>"""

    # ── Ratio rows ───────────────────────────────────────────────────────────
    ratio_rows = ""
    for rr in scoring.ratios:
        ratio_rows += f"""
        <tr>
          <td style="padding:5px 12px;border-bottom:1px solid #F9FAFB;color:#374151;font-size:12px">{rr.name.replace('_',' ').title()}</td>
          <td style="padding:5px 12px;border-bottom:1px solid #F9FAFB;font-size:12px;font-weight:600;color:#0F172A;text-align:center">{rr.display_value}</td>
          <td style="padding:5px 12px;border-bottom:1px solid #F9FAFB;text-align:center">{_signal_badge(rr.signal)}</td>
          <td style="padding:5px 12px;border-bottom:1px solid #F9FAFB;font-size:11px;color:#6B7280">{rr.normalised_score:.0f}/100</td>
        </tr>"""

    # ── Risk flags ───────────────────────────────────────────────────────────
    all_flags  = list(scoring.red_flags) + ai.key_risks[:3]
    flag_items = ""
    for f in all_flags[:8]:
        flag_items += (
            f'<div style="background:#FEF2F2;border-left:4px solid #DC2626;'
            f'padding:7px 12px;border-radius:5px;margin:4px 0;font-size:12px;color:#991B1B">'
            f'⚠️ &nbsp;{f}</div>'
        )
    if not flag_items:
        flag_items = (
            '<div style="background:#D1FAE5;border-left:4px solid #16A34A;'
            'padding:7px 12px;border-radius:5px;font-size:12px;color:#065F46">'
            '✅ &nbsp;No significant risk flags identified.</div>'
        )

    # ── Positive signals ─────────────────────────────────────────────────────
    all_pos   = list(scoring.positive_signals)[:4] + ai.positive_factors[:3]
    pos_items = ""
    for p in all_pos[:6]:
        pos_items += (
            f'<div style="background:#D1FAE5;border-left:4px solid #16A34A;'
            f'padding:7px 12px;border-radius:5px;margin:4px 0;font-size:12px;color:#065F46">'
            f'✅ &nbsp;{p}</div>'
        )
    if not pos_items:
        pos_items = '<div style="color:#6B7280;font-size:12px">No positive signals available.</div>'

    # ── Conditions ───────────────────────────────────────────────────────────
    all_conds  = list(decision.conditions) + ai.suggested_conditions[:4]
    cond_items = ""
    for c in all_conds[:8]:
        cond_items += (
            f'<div style="background:#DBEAFE;border-left:4px solid #2274E0;'
            f'padding:7px 12px;border-radius:5px;margin:4px 0;font-size:12px;color:#1E40AF">'
            f'📋 &nbsp;{c}</div>'
        )
    if not cond_items:
        cond_items = '<div style="color:#6B7280;font-size:12px">No specific conditions attached.</div>'

    # ── AI narrative ─────────────────────────────────────────────────────────
    narrative_lines = [
        ln.strip().lstrip("•").strip()
        for ln in ai.risk_narrative.split("\n")
        if ln.strip().lstrip("•").strip()
    ][:6]
    narrative_html = "".join(
        f'<div style="background:#F0F7FF;border-left:3px solid #2274E0;'
        f'padding:6px 12px;border-radius:4px;margin:3px 0;font-size:12px;color:#1E3A5F">'
        f'• {line}</div>'
        for line in narrative_lines
    ) or '<div style="color:#6B7280;font-size:12px">No AI narrative available.</div>'

    # ── Key financials ───────────────────────────────────────────────────────
    fin_rows = [
        ("Annual Revenue",      financials.get("revenue")),
        ("EBITDA",              financials.get("ebitda")),
        ("Net Profit",          financials.get("net_profit")),
        ("Total Debt",          financials.get("total_debt")),
        ("Equity",              financials.get("equity")),
        ("Operating Cash Flow", financials.get("operating_cash_flow")),
    ]
    fin_items = ""
    for label, val in fin_rows:
        if val is not None:
            fin_items += f"""
            <tr>
              <td style="padding:5px 12px;border-bottom:1px solid #F9FAFB;color:#374151;font-size:12px">{label}</td>
              <td style="padding:5px 12px;border-bottom:1px solid #F9FAFB;font-weight:700;color:#0F172A;font-size:12px;text-align:right">{_fmt_inr(val)}</td>
            </tr>"""

    # ── External sentiment ───────────────────────────────────────────────────
    sentiment      = ext.get("sentiment", "N/A")
    sent_color_map = {"Positive": "#16A34A", "Neutral": "#D97706", "Negative": "#DC2626"}
    sent_color_val = sent_color_map.get(sentiment, "#475569")
    sent_dot       = {"Positive": "🟢", "Neutral": "🟡", "Negative": "🔴"}.get(sentiment, "⚪")

    # ── DSO hint ─────────────────────────────────────────────────────────────
    dso_str = f"{decision.dso_days:.0f} days" if decision.dso_days else "N/A"

    # ════════════════════════════════════════════════════════════════════════
    # HTML BODY
    # ════════════════════════════════════════════════════════════════════════
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{subject}</title>
</head>
<body style="margin:0;padding:0;background:#F1F5F9;font-family:Arial,Helvetica,sans-serif">

<!-- wrapper -->
<table width="100%" cellpadding="0" cellspacing="0" style="background:#F1F5F9;padding:24px 0">
<tr><td align="center">
<table width="680" cellpadding="0" cellspacing="0" style="max-width:680px;width:100%">

  <!-- ── HEADER ──────────────────────────────────────────────────── -->
  <tr>
    <td style="background:#0F172A;border-radius:12px 12px 0 0;padding:22px 28px">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td>
            <div style="color:#94A3B8;font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:0.1em">TravelPlus · Credit Evaluation Report</div>
            <div style="color:#F8FAFC;font-size:20px;font-weight:800;margin:4px 0">{company}</div>
            <div style="color:#94A3B8;font-size:12px">Prepared: {eval_date}</div>
          </td>
          <td align="right" style="vertical-align:top">
            <div style="background:{decision_bg};color:{decision_color};font-weight:800;font-size:15px;
                        border-radius:24px;padding:7px 22px;display:inline-block;border:2px solid {decision_color}">
              {('✅' if approved else '❌')} &nbsp;{decision_label}
            </div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- ── SCORE STRIP ──────────────────────────────────────────────── -->
  <tr>
    <td style="background:#1E293B;padding:14px 28px">
      <table width="100%" cellpadding="0" cellspacing="0">
        <tr>
          <td style="text-align:center;padding:0 12px">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;text-transform:uppercase">Composite Score</div>
            <div style="color:#F8FAFC;font-size:28px;font-weight:800">{score:.0f}<span style="font-size:14px;color:#94A3B8">/100</span></div>
            <div style="background:{band_bg};color:{band_fg};border-radius:20px;padding:2px 12px;font-size:11px;font-weight:700;display:inline-block;margin-top:3px">{band} Risk</div>
          </td>
          <td style="background:#334155;width:1px"></td>
          <td style="text-align:center;padding:0 12px">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;text-transform:uppercase">Credit Limit</div>
            <div style="color:#2274E0;font-size:22px;font-weight:800">{_fmt_inr(decision.credit_limit)}</div>
          </td>
          <td style="background:#334155;width:1px"></td>
          <td style="text-align:center;padding:0 12px">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;text-transform:uppercase">Payment Terms</div>
            <div style="color:#F8FAFC;font-size:22px;font-weight:800">Net-{decision.payment_terms_days}</div>
          </td>
          <td style="background:#334155;width:1px"></td>
          <td style="text-align:center;padding:0 12px">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;text-transform:uppercase">Max Exposure</div>
            <div style="color:#F8FAFC;font-size:22px;font-weight:800">{_fmt_inr(decision.max_safe_exposure)}</div>
          </td>
          <td style="background:#334155;width:1px"></td>
          <td style="text-align:center;padding:0 12px">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;text-transform:uppercase">AI Decision</div>
            <div style="color:{ai_color};font-size:14px;font-weight:800">{ai_rec}</div>
            <div style="color:#94A3B8;font-size:10px">{ai.confidence_level} ({ai.confidence_score:.0%})</div>
          </td>
          <td style="background:#334155;width:1px"></td>
          <td style="text-align:center;padding:0 12px">
            <div style="color:#94A3B8;font-size:10px;font-weight:700;text-transform:uppercase">News Sentiment</div>
            <div style="color:{sent_color_val};font-size:14px;font-weight:800">{sent_dot} {sentiment}</div>
          </td>
        </tr>
      </table>
    </td>
  </tr>

  <!-- ── MAIN CONTENT AREA ─────────────────────────────────────────── -->
  <tr>
    <td style="background:#FFFFFF;border-radius:0 0 12px 12px;padding:24px 28px">

      <!-- ── SECTION: Score Breakdown ──────────────────────────────── -->
      <div style="background:#DBEAFE;border-left:4px solid #2274E0;border-radius:6px;padding:7px 14px;margin-bottom:12px">
        <span style="font-size:12px;font-weight:800;color:#0F172A;text-transform:uppercase;letter-spacing:0.05em">📊 Score Breakdown</span>
      </div>
      <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #E2E8F0;border-radius:8px;overflow:hidden;margin-bottom:20px">
        <thead>
          <tr style="background:#0F172A">
            <th style="padding:8px 12px;text-align:left;color:#F8FAFC;font-size:11px;font-weight:700">Sub-Model</th>
            <th style="padding:8px 12px;text-align:center;color:#F8FAFC;font-size:11px;font-weight:700">Weight</th>
            <th style="padding:8px 12px;text-align:left;color:#F8FAFC;font-size:11px;font-weight:700">Score</th>
          </tr>
        </thead>
        <tbody>{sub_rows}</tbody>
        <tfoot>
          <tr style="background:#F8FAFC">
            <td colspan="2" style="padding:8px 12px;font-weight:800;font-size:13px;color:#0F172A">Composite Score</td>
            <td style="padding:8px 12px;font-weight:800;font-size:13px;color:#2274E0">{score:.1f} / 100</td>
          </tr>
        </tfoot>
      </table>

      <!-- ── TWO COLUMNS: Financials | Ratios ────────────────────── -->
      <table width="100%" cellpadding="0" cellspacing="0" style="margin-bottom:20px">
        <tr>
          <!-- Key Financials -->
          <td width="48%" style="vertical-align:top;padding-right:10px">
            <div style="background:#DBEAFE;border-left:4px solid #2274E0;border-radius:6px;padding:7px 14px;margin-bottom:10px">
              <span style="font-size:12px;font-weight:800;color:#0F172A;text-transform:uppercase;letter-spacing:0.05em">💰 Key Financials</span>
            </div>
            <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #E2E8F0;border-radius:8px;overflow:hidden">
              <thead>
                <tr style="background:#F8FAFC">
                  <th style="padding:6px 12px;text-align:left;color:#374151;font-size:11px;font-weight:700">Metric</th>
                  <th style="padding:6px 12px;text-align:right;color:#374151;font-size:11px;font-weight:700">Value</th>
                </tr>
              </thead>
              <tbody>{fin_items}</tbody>
            </table>
          </td>
          <td width="4%"></td>
          <!-- Financial Ratios -->
          <td width="48%" style="vertical-align:top;padding-left:10px">
            <div style="background:#DBEAFE;border-left:4px solid #2274E0;border-radius:6px;padding:7px 14px;margin-bottom:10px">
              <span style="font-size:12px;font-weight:800;color:#0F172A;text-transform:uppercase;letter-spacing:0.05em">📐 Financial Ratios</span>
            </div>
            <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #E2E8F0;border-radius:8px;overflow:hidden">
              <thead>
                <tr style="background:#F8FAFC">
                  <th style="padding:6px 12px;text-align:left;color:#374151;font-size:11px;font-weight:700">Ratio</th>
                  <th style="padding:6px 12px;text-align:center;color:#374151;font-size:11px;font-weight:700">Value</th>
                  <th style="padding:6px 12px;text-align:center;color:#374151;font-size:11px;font-weight:700">Signal</th>
                  <th style="padding:6px 12px;text-align:left;color:#374151;font-size:11px;font-weight:700">Score</th>
                </tr>
              </thead>
              <tbody>{ratio_rows}</tbody>
            </table>
          </td>
        </tr>
      </table>

      <!-- ── SECTION: Risk Flags ─────────────────────────────────── -->
      <div style="background:#FEE2E2;border-left:4px solid #DC2626;border-radius:6px;padding:7px 14px;margin-bottom:10px">
        <span style="font-size:12px;font-weight:800;color:#991B1B;text-transform:uppercase;letter-spacing:0.05em">🚩 Risk Flags &amp; Key Risks</span>
      </div>
      <div style="margin-bottom:20px">{flag_items}</div>

      <!-- ── SECTION: Positive Signals ──────────────────────────── -->
      <div style="background:#D1FAE5;border-left:4px solid #16A34A;border-radius:6px;padding:7px 14px;margin-bottom:10px">
        <span style="font-size:12px;font-weight:800;color:#065F46;text-transform:uppercase;letter-spacing:0.05em">✅ Positive Signals</span>
      </div>
      <div style="margin-bottom:20px">{pos_items}</div>

      <!-- ── SECTION: AI Assessment ──────────────────────────────── -->
      <div style="background:#DBEAFE;border-left:4px solid #2274E0;border-radius:6px;padding:7px 14px;margin-bottom:10px">
        <span style="font-size:12px;font-weight:800;color:#0F172A;text-transform:uppercase;letter-spacing:0.05em">🤖 AI Assessment — {ai_rec} ({ai.confidence_level})</span>
      </div>
      <div style="margin-bottom:20px">{narrative_html}</div>

      <!-- ── SECTION: Credit Terms & Conditions ─────────────────── -->
      <div style="background:#DBEAFE;border-left:4px solid #2274E0;border-radius:6px;padding:7px 14px;margin-bottom:10px">
        <span style="font-size:12px;font-weight:800;color:#0F172A;text-transform:uppercase;letter-spacing:0.05em">📋 Credit Terms &amp; Conditions</span>
      </div>
      <table width="100%" cellpadding="0" cellspacing="0" style="border:1px solid #E2E8F0;border-radius:8px;overflow:hidden;margin-bottom:14px">
        <tr style="background:#F8FAFC">
          <td style="padding:10px 16px;font-size:12px;color:#374151;border-right:1px solid #E2E8F0"><b>Credit Limit</b><br><span style="font-size:16px;font-weight:800;color:#2274E0">{_fmt_inr(decision.credit_limit)}</span></td>
          <td style="padding:10px 16px;font-size:12px;color:#374151;border-right:1px solid #E2E8F0"><b>Max Safe Exposure</b><br><span style="font-size:16px;font-weight:800;color:#0F172A">{_fmt_inr(decision.max_safe_exposure)}</span></td>
          <td style="padding:10px 16px;font-size:12px;color:#374151;border-right:1px solid #E2E8F0"><b>Payment Terms</b><br><span style="font-size:16px;font-weight:800;color:#0F172A">Net-{decision.payment_terms_days} days</span></td>
          <td style="padding:10px 16px;font-size:12px;color:#374151"><b>DSO</b><br><span style="font-size:16px;font-weight:800;color:#0F172A">{dso_str}</span></td>
        </tr>
      </table>
      <div style="margin-bottom:20px">{cond_items}</div>

      <!-- ── DIVIDER ────────────────────────────────────────────── -->
      <hr style="border:none;border-top:1px solid #E2E8F0;margin:16px 0">

      <!-- ── RATIONALE (collapsible note) ──────────────────────── -->
      <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:12px 16px;margin-bottom:20px">
        <div style="font-size:11px;font-weight:700;color:#0F172A;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:6px">📊 Limit Calculation Rationale</div>
        <div style="font-size:12px;color:#374151;line-height:1.6">{decision.rationale}</div>
      </div>

    </td>
  </tr>

  <!-- ── FOOTER ─────────────────────────────────────────────────────── -->
  <tr>
    <td style="padding:16px 28px 0 28px">
      <div style="background:#0F172A;border-radius:8px;padding:14px 20px">
        <div style="color:#94A3B8;font-size:11px;line-height:1.6">
          This report was generated automatically by the <strong style="color:#CBD5E1">TravelPlus Credit Evaluation Dashboard</strong> on {eval_date}.
          The analysis combines quantitative financial scoring with AI-assisted assessment.
          All credit decisions are subject to final review by the authorised credit officer.
          <br><br>
          <span style="color:#64748B">Confidential — for internal use only. Do not distribute externally.</span>
        </div>
      </div>
    </td>
  </tr>

</table>
</td></tr>
</table>

</body>
</html>"""

    return subject, html


# ---------------------------------------------------------------------------
# SMTP sender
# ---------------------------------------------------------------------------

def _build_mime(subject, sender_address, recipients, html_body) -> MIMEMultipart:
    """Assemble the MIME message."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = f"TravelPlus Credit Team <{sender_address}>"
    msg["To"]      = ", ".join(recipients)
    plain = (
        "Credit Evaluation Report\n\n"
        "This email contains an HTML credit evaluation summary. "
        "Please open it in an HTML-capable email client.\n"
    )
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_body, "html"))
    return msg


def send_email(
    recipients: List[str],
    subject: str,
    html_body: str,
    smtp_host: str,
    smtp_port: int,
    smtp_user: str,
    smtp_password: str,
    sender_address: str,
) -> Tuple[bool, str]:
    """
    Send an HTML email.  Tries STARTTLS (port 587) first; if the caller
    passed port 465 it uses SMTP_SSL instead.

    Returns
    -------
    (success: bool, message: str)
    """
    if not smtp_user or not smtp_password:
        return False, (
            "SMTP credentials not configured. "
            "Add SMTP_USER and SMTP_PASSWORD to your .env file."
        )
    if not recipients:
        return False, "No recipients specified."

    msg = _build_mime(subject, sender_address, recipients, html_body)
    context = ssl.create_default_context()

    try:
        if smtp_port == 465:
            # SSL from the start (less common today)
            with smtplib.SMTP_SSL(smtp_host, smtp_port, context=context, timeout=15) as server:
                server.login(smtp_user, smtp_password)
                server.sendmail(sender_address, recipients, msg.as_string())
        else:
            # STARTTLS — standard for port 587
            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls(context=context)
                server.ehlo()
                server.login(smtp_user, smtp_password)
                server.sendmail(sender_address, recipients, msg.as_string())

        log.info("Email sent to %d recipients: %s", len(recipients), recipients)
        return True, f"Email sent successfully to {len(recipients)} recipient(s)."

    except smtplib.SMTPAuthenticationError:
        hint = (
            "Authentication failed. "
            "If 2-Step Verification is enabled on this Google Workspace account, "
            "you need an App Password instead of the regular password. "
            "Go to: myaccount.google.com → Security → 2-Step Verification → App Passwords."
        )
        log.error(hint)
        return False, hint
    except smtplib.SMTPConnectError as exc:
        msg_err = f"Cannot connect to {smtp_host}:{smtp_port} — {exc}"
        log.error(msg_err)
        return False, msg_err
    except smtplib.SMTPRecipientsRefused as exc:
        msg_err = f"One or more recipients refused: {exc}"
        log.error(msg_err)
        return False, msg_err
    except OSError as exc:
        # Network-level errors (DNS, timeout, etc.)
        msg_err = f"Network error while sending email: {exc}"
        log.error(msg_err)
        return False, msg_err
    except Exception as exc:
        msg_err = f"Unexpected email error: {exc}"
        log.exception(msg_err)
        return False, msg_err


def send_result(result: dict, recipients: List[str]) -> Tuple[bool, str]:
    """
    Build and send a CFO email for a completed evaluation result dict.
    Reads SMTP config from config.py / environment.
    """
    from config import (
        SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, SMTP_FROM,
    )
    subject, html = build_cfo_email(result)
    return send_email(
        recipients    = recipients,
        subject       = subject,
        html_body     = html,
        smtp_host     = SMTP_HOST,
        smtp_port     = SMTP_PORT,
        smtp_user     = SMTP_USER,
        smtp_password = SMTP_PASSWORD,
        sender_address= SMTP_FROM,
    )
