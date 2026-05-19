# Credit Evaluation Dashboard — Setup Guide

## Quick Start (3 commands)

```bash
cd C:\Users\Inno\credit_dashboard
pip install -r requirements.txt
streamlit run app.py
```

The app opens at http://localhost:8501

---

## Step-by-Step

### 1. Prerequisites
- Python 3.10+
- pip

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
copy .env.example .env
```

Edit `.env`:

```
# Required for AI analysis (pick one):
ANTHROPIC_API_KEY=sk-ant-your-key-here
# OR
# OPENAI_API_KEY=sk-your-key-here
# LLM_PROVIDER=openai

# Optional — for live news (falls back to DuckDuckGo scrape if omitted):
NEWS_API_KEY=your-newsapi-org-key
```

> **No API key?** The system runs fully in **mock mode** — all AI and news
> responses are deterministic templates based on the credit score.
> You can demo end-to-end without any keys.

### 4. Run the app

```bash
streamlit run app.py
```

---

## Usage

### Option A — Demo (no file needed)
Click one of the three demo buttons on the welcome screen:
- **GOOD Company** — strong financials, clean history
- **AVERAGE Company** — mixed signals, conditional approval
- **RISKY Company** — weak liquidity, rejection scenario

### Option B — CSV Upload
Upload `sample_data/sample_financials.csv` (long format) or
`sample_data/sample_financials_wide.csv` (wide format).

### Option C — Manual Entry
Fill in all fields in the sidebar "Manual Entry" panel.

### Option D — PDF Upload
Upload any PDF financial statement. The parser extracts text
via pdfplumber table detection + regex fallback.

---

## Configuration

All tunable parameters are in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SCORING_WEIGHTS` | 40/30/20/10 | Sub-model weights |
| `RATIO_BENCHMARKS` | See file | Per-ratio good/acceptable/critical thresholds |
| `CREDIT_LIMIT_BASE_PCT` | 25% | % of monthly revenue as base limit |
| `PAYMENT_TERMS_MAP` | 60/30/14 days | Net days by risk band |
| `NEWS_LOOKBACK_DAYS` | 90 | News search window |

---

## Project Structure

```
credit_dashboard/
├── app.py              # Streamlit UI (entry point)
├── config.py           # All weights, thresholds, API keys
├── utils.py            # Logging, formatters, helpers
├── database.py         # SQLite persistence
├── ingestion.py        # File upload routing
├── parser.py           # PDF / CSV / Excel extraction
├── scoring.py          # Weighted scoring model (Model 1)
├── ai_engine.py        # LLM reasoning engine (Model 2)
├── external_data.py    # News / risk signals
├── credit_engine.py    # Dynamic credit limit engine
├── load_env.py         # .env auto-loader
├── sample_data/
│   ├── sample_financials.csv        # Long-format sample
│   ├── sample_financials_wide.csv   # Wide-format sample
│   └── sample_payment_history.csv   # Payment ageing sample
├── logs/               # Auto-created log files
├── exports/            # Reserved for future PDF exports
├── requirements.txt
└── .env.example
```

---

## Production Deployment

### Docker (recommended)

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t credit-dashboard .
docker run -p 8501:8501 --env-file .env credit-dashboard
```

### Streamlit Community Cloud

1. Push to GitHub
2. Go to share.streamlit.io
3. Set secrets (API keys) in the Secrets panel
4. Deploy

### Upgrade SQLite → PostgreSQL

Replace `database.py` connection string with `psycopg2`:

```python
import psycopg2
con = psycopg2.connect(os.getenv("DATABASE_URL"))
```

---

## Extending the Scoring Model

To add a new financial ratio:

1. Add a `RatioBenchmark` entry to `RATIO_BENCHMARKS` in `config.py`
2. Compute the ratio value in `scoring.py::compute_ratios()`
3. The rest of the pipeline picks it up automatically

To change weights:

```python
# config.py
SCORING_WEIGHTS = {
    "financial_health":      0.45,  # was 0.40
    "payment_behavior":      0.25,  # was 0.30
    "external_risk":         0.20,
    "operational_stability": 0.10,
}
```
