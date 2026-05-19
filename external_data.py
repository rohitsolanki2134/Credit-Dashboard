"""
External Risk Engine: fetches and analyses news/web signals for a company.

Priority:
  1. NewsAPI (if NEWS_API_KEY set)
  2. SerpAPI (if SERP_API_KEY set)
  3. Google News RSS fallback (no key required)
  4. Mock data (demo / CI)

Recency policy:
  - Articles within the last 12 months are always included.
  - Articles older than 12 months are included ONLY if they contain
    evergreen financial keywords (funding raised, merger/acquisition, IPO).
  - All queries are scoped to financial topics so general PR noise is
    filtered out at the source.

Output schema:
  {
    "sentiment": "Positive" | "Neutral" | "Negative",
    "risk_flags": [...],
    "positive_signals": [...],
    "summary": "...",
    "articles": [{"title": ..., "source": ..., "date": ..., "url": ..., "sentiment": ...}],
    "source": "newsapi" | "serpapi" | "google_news" | "mock",
  }
"""

from __future__ import annotations
import re
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from urllib.parse import quote_plus

from config import (
    NEWS_API_KEY,
    SERP_API_KEY,
    NEWS_LOOKBACK_DAYS,
    NEWS_MAX_ARTICLES,
    NEGATIVE_KEYWORDS,
    POSITIVE_KEYWORDS,
    EVERGREEN_NEWS_KEYWORDS,
    FINANCIAL_SEARCH_TERMS,
)
from utils import get_logger

log = get_logger(__name__)

ExternalResult = Dict[str, Any]

# Hard ceiling: never return articles older than this many days regardless of content
_MAX_AGE_DAYS = 365
# Evergreen exception: funding / M&A news older than 12 months is still credit-relevant
_EVERGREEN_MAX_AGE_DAYS = 365 * 3   # keep up to 3 years for funding / M&A context


# ---------------------------------------------------------------------------
# Date parsing helper
# ---------------------------------------------------------------------------

def _parse_date(date_str: str) -> Optional[datetime]:
    """Try several common date formats; return None if unparseable."""
    if not date_str:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z",
                "%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"):
        try:
            dt = datetime.strptime(date_str[:len(fmt) + 5].strip(), fmt)
            # Make timezone-naive for comparison
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except ValueError:
            continue
    # email.utils fallback (handles RFC 2822 like NewsAPI/RSS)
    try:
        from email.utils import parsedate_to_datetime
        return parsedate_to_datetime(date_str).replace(tzinfo=None)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Recency filter
# ---------------------------------------------------------------------------

def _is_article_relevant(article: dict) -> bool:
    """
    Return True if the article should be included.

    Rules:
      - If published within the last 12 months  → always include.
      - If published 1–3 years ago AND the title/description contains
        an evergreen keyword (funding, merger, acquisition, IPO, …) → include.
      - Otherwise → exclude.
    """
    date_str = article.get("date", "")
    pub_date = _parse_date(date_str)
    now = datetime.utcnow()

    if pub_date is None:
        # Unknown date — include by default (better than silently dropping)
        return True

    age_days = (now - pub_date).days

    # Within standard window
    if age_days <= _MAX_AGE_DAYS:
        return True

    # Older than 1 year — only keep if it's evergreen financial content
    if age_days <= _EVERGREEN_MAX_AGE_DAYS:
        text = f"{article.get('title', '')} {article.get('description', '')}".lower()
        if any(kw in text for kw in EVERGREEN_NEWS_KEYWORDS):
            log.debug(
                "Keeping older article (%d days) due to evergreen keyword: %s",
                age_days, article.get("title", "")[:60],
            )
            return True

    return False


# ---------------------------------------------------------------------------
# Sentiment classifier (keyword-based)
# ---------------------------------------------------------------------------

def _score_text(text: str) -> int:
    """Return +1 (positive), -1 (negative), 0 (neutral) for a text snippet."""
    text_lower = text.lower()
    neg = sum(1 for kw in NEGATIVE_KEYWORDS if kw in text_lower)
    pos = sum(1 for kw in POSITIVE_KEYWORDS if kw in text_lower)
    if neg > pos:
        return -1
    if pos > neg:
        return 1
    return 0


def _classify_articles(
    articles: List[dict],
) -> tuple[str, List[str], List[str], List[dict]]:
    """
    Classify sentiment across all articles and extract flag keywords.

    Returns:
      overall_sentiment  – "Positive" | "Neutral" | "Negative"
      risk_flags         – list of flagged keyword strings  (backward-compat)
      positive_signals   – list of positive keyword strings
      risk_flags_detail  – list of {keyword, label, articles:[{title,url,source,date}]}
                           one entry per unique keyword, each with the articles that
                           triggered it so the UI can render clickable source links.
    """
    scores: List[int] = []
    # keyword (lower) → {"label": ..., "articles": [...]}
    flag_map: Dict[str, dict] = {}
    pos_signals: List[str] = []

    for art in articles:
        text = f"{art.get('title', '')} {art.get('description', '')}".lower()
        s = _score_text(text)
        scores.append(s)

        art_ref = {
            "title":  art.get("title", ""),
            "url":    art.get("url", ""),
            "source": art.get("source", ""),
            "date":   (art.get("date") or "")[:10],
        }

        for kw in NEGATIVE_KEYWORDS:
            if kw in text:
                label = kw.title()
                if kw not in flag_map:
                    flag_map[kw] = {"label": label, "articles": []}
                # Avoid duplicate article refs for the same keyword
                if art_ref not in flag_map[kw]["articles"]:
                    flag_map[kw]["articles"].append(art_ref)

        for kw in POSITIVE_KEYWORDS:
            if kw in text and kw.title() not in pos_signals:
                pos_signals.append(kw.title())

    if not scores:
        return "Neutral", [], [], []

    avg = sum(scores) / len(scores)
    if avg > 0.1:
        overall = "Positive"
    elif avg < -0.1:
        overall = "Negative"
    else:
        overall = "Neutral"

    # Build ordered outputs — cap at 8 risk flags
    flag_items = list(flag_map.values())[:8]
    risk_flags = [f["label"] for f in flag_items]
    risk_flags_detail = flag_items  # full detail for UI rendering

    return overall, risk_flags, pos_signals[:5], risk_flags_detail


def _build_summary(articles: List[dict], sentiment: str, flags: List[str]) -> str:
    if not articles:
        return "No recent financial news found for this company."

    lines = [f"Overall sentiment: **{sentiment}**. Recent financial headlines:"]
    for art in articles[:5]:
        title  = art.get("title", "No title")
        source = art.get("source", "Unknown")
        date   = art.get("date", "")[:10]
        lines.append(f"• {title} — {source} ({date})")

    if flags:
        lines.append(f"\nRisk keywords detected: {', '.join(flags[:5])}")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Provider: NewsAPI
# ---------------------------------------------------------------------------

def _fetch_newsapi(company_name: str) -> List[dict]:
    import requests

    # Fetch up to 1 year; recency filter applied post-fetch
    from_date = (datetime.utcnow() - timedelta(days=_MAX_AGE_DAYS)).strftime("%Y-%m-%d")

    # Financially scoped query — reduces PR noise
    query = f'"{company_name}" AND ({FINANCIAL_SEARCH_TERMS})'

    url = "https://newsapi.org/v2/everything"
    params = {
        "q":        query,
        "from":     from_date,
        "sortBy":   "relevancy",
        "pageSize": NEWS_MAX_ARTICLES,
        "language": "en",
        "apiKey":   NEWS_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    articles = []
    for a in data.get("articles", []):
        articles.append({
            "title":       a.get("title", ""),
            "description": a.get("description", ""),
            "source":      a.get("source", {}).get("name", ""),
            "date":        a.get("publishedAt", ""),
            "url":         a.get("url", ""),
        })
    return articles


# ---------------------------------------------------------------------------
# Provider: SerpAPI
# ---------------------------------------------------------------------------

def _fetch_serpapi(company_name: str) -> List[dict]:
    import requests

    # Financially focused query
    query = f"{company_name} financial results earnings revenue funding merger"

    url = "https://serpapi.com/search"
    params = {
        "q":       query,
        "tbm":     "nws",
        "num":     NEWS_MAX_ARTICLES,
        "api_key": SERP_API_KEY,
    }
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json()

    articles = []
    for r in data.get("news_results", []):
        articles.append({
            "title":       r.get("title", ""),
            "description": r.get("snippet", ""),
            "source":      r.get("source", ""),
            "date":        r.get("date", ""),
            "url":         r.get("link", ""),
        })
    return articles


# ---------------------------------------------------------------------------
# Provider: Google News RSS (no API key required)
# ---------------------------------------------------------------------------

def _fetch_google_news_rss(company_name: str) -> List[dict]:
    """
    Fetch financial news via Google News RSS.
    Query is scoped to financial topics to avoid general PR noise.
    Falls back gracefully; returns [] on any error.
    """
    try:
        import requests
        import xml.etree.ElementTree as ET
    except ImportError:
        log.warning("requests not available; cannot fetch Google News RSS")
        return []

    # Financially scoped search: company name + financial context terms
    search_query = (
        f'"{company_name}" '
        f'(earnings OR revenue OR profit OR funding OR merger OR acquisition '
        f'OR results OR debt OR IPO OR quarterly OR "annual results" OR "financial results")'
    )
    query = quote_plus(search_query)
    url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; CreditDashboard/1.0)"}

    try:
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        log.warning("Google News RSS fetch failed: %s", e)
        return []

    articles = []
    try:
        root = ET.fromstring(resp.content)
        channel = root.find("channel")
        if channel is None:
            return []

        for item in channel.findall("item")[:NEWS_MAX_ARTICLES * 2]:   # fetch extra; filter below
            title     = (item.findtext("title") or "").strip()
            link      = (item.findtext("link")  or "").strip()
            pub       = (item.findtext("pubDate") or "").strip()
            source_el = item.find("source")
            source    = source_el.text.strip() if source_el is not None else ""
            desc_raw  = (item.findtext("description") or "").strip()
            desc      = re.sub(r"<[^>]+>", " ", desc_raw).strip()

            # Normalise date to YYYY-MM-DD
            try:
                from email.utils import parsedate_to_datetime
                pub_date = parsedate_to_datetime(pub).strftime("%Y-%m-%d")
            except Exception:
                pub_date = datetime.utcnow().strftime("%Y-%m-%d")

            if title:
                articles.append({
                    "title":       title,
                    "description": desc,
                    "source":      source,
                    "date":        pub_date,
                    "url":         link,
                })
    except ET.ParseError as e:
        log.warning("Google News RSS XML parse error: %s", e)

    log.info("Google News RSS returned %d articles for '%s'", len(articles), company_name)
    return articles


# ---------------------------------------------------------------------------
# Provider: Mock (demo / CI)
# ---------------------------------------------------------------------------

def _mock_result(company_name: str) -> ExternalResult:
    now = datetime.utcnow()
    articles = [
        {
            "title":       f"{company_name} Reports Record Q3 Revenue Growth",
            "description": "The company posted strong revenue growth beating analyst expectations.",
            "source": "Reuters",
            "date":   (now - timedelta(days=30)).strftime("%Y-%m-%d"),
            "url": "",
        },
        {
            "title":       f"{company_name} Raises ₹500 Cr in Series C Funding Round",
            "description": "New capital will be used for technology infrastructure and market expansion.",
            "source": "Economic Times",
            "date":   (now - timedelta(days=200)).strftime("%Y-%m-%d"),
            "url": "",
        },
        {
            "title":       f"{company_name} Expands into Southeast Asia via Acquisition",
            "description": "New strategic acquisition announced for regional market entry.",
            "source": "Bloomberg",
            "date":   (now - timedelta(days=90)).strftime("%Y-%m-%d"),
            "url": "",
        },
        {
            "title":       f"{company_name} Management Discusses Cost Restructuring",
            "description": "The company is reviewing operational costs amid margin pressure.",
            "source": "FT",
            "date":   (now - timedelta(days=60)).strftime("%Y-%m-%d"),
            "url": "",
        },
    ]
    sentiment, flags, positives, flags_detail = _classify_articles(articles)
    return {
        "sentiment":          sentiment,
        "risk_flags":         flags,
        "risk_flags_detail":  flags_detail,
        "positive_signals":   positives,
        "summary":            _build_summary(articles, sentiment, flags),
        "articles":           articles,
        "source":             "mock",
        "fetched_at":         now.isoformat(),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_external_risk(company_name: str, use_mock: bool = False) -> ExternalResult:
    """
    Fetch and analyse external financial risk signals for a company.

    Provider priority: NewsAPI → SerpAPI → Google News RSS → mock fallback.

    Recency policy applied after fetching:
      - Articles ≤ 12 months old: always included.
      - Articles > 12 months old: included only if they contain evergreen
        financial keywords (funding raised, M&A, IPO).
      - Articles > 3 years old: always excluded.
    """
    if use_mock:
        log.info("Mock news requested for '%s'", company_name)
        return _mock_result(company_name)

    articles: List[dict] = []
    source = "unknown"

    if NEWS_API_KEY:
        try:
            articles = _fetch_newsapi(company_name)
            source = "newsapi"
            log.info("NewsAPI returned %d articles", len(articles))
        except Exception as e:
            log.warning("NewsAPI failed: %s — trying SerpAPI", e)

    if not articles and SERP_API_KEY:
        try:
            articles = _fetch_serpapi(company_name)
            source = "serpapi"
            log.info("SerpAPI returned %d articles", len(articles))
        except Exception as e:
            log.warning("SerpAPI failed: %s — trying Google News RSS", e)

    if not articles:
        try:
            articles = _fetch_google_news_rss(company_name)
            source = "google_news"
        except Exception as e:
            log.warning("Google News RSS failed: %s", e)

    _empty = {
        "sentiment":         "Neutral",
        "risk_flags":        [],
        "risk_flags_detail": [],
        "positive_signals":  [],
        "articles":          [],
        "fetched_at":        datetime.utcnow().isoformat(),
    }

    if not articles:
        log.warning("No external data found for '%s' — returning empty result", company_name)
        return {**_empty,
                "summary": "No recent financial news found for this company. Sentiment scored as neutral.",
                "source":  "none"}

    # ── Apply recency filter ─────────────────────────────────────────────────
    before_filter = len(articles)
    articles = [a for a in articles if _is_article_relevant(a)]
    dropped = before_filter - len(articles)
    if dropped:
        log.info(
            "Recency filter: dropped %d article(s) older than 12 months "
            "(no evergreen keywords) for '%s'",
            dropped, company_name,
        )

    # Cap at configured maximum
    articles = articles[:NEWS_MAX_ARTICLES]

    if not articles:
        return {**_empty,
                "summary": "No financial news within the last 12 months found for this company.",
                "source":  source}

    sentiment, flags, positives, flags_detail = _classify_articles(articles)

    return {
        "sentiment":          sentiment,
        "risk_flags":         flags,
        "risk_flags_detail":  flags_detail,
        "positive_signals":   positives,
        "summary":            _build_summary(articles, sentiment, flags),
        "articles":           articles,
        "source":             source,
        "fetched_at":         datetime.utcnow().isoformat(),
    }
