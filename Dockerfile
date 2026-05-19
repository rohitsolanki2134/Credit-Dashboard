# ─────────────────────────────────────────────────────────────────────────────
# TravelPlus Credit Dashboard — Google Cloud Run Dockerfile
# ─────────────────────────────────────────────────────────────────────────────
# Build & push:
#   gcloud builds submit --config cloudbuild.yaml
#
# Run locally to test the container:
#   docker build -t credit-dashboard .
#   docker run -p 8080:8080 --env-file .env credit-dashboard
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

# ── System deps ───────────────────────────────────────────────────────────────
# curl  : used by the HEALTHCHECK
# libgomp1 : required by PyMuPDF (PDF parser)
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Python deps (cached layer — only rebuilds when requirements.txt changes) ──
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ──────────────────────────────────────────────────────────
COPY . .

# ── Runtime configuration ─────────────────────────────────────────────────────
# Cloud Run injects PORT (default 8080). Streamlit must bind to 0.0.0.0:$PORT.
ENV PORT=8080
EXPOSE 8080

# ── Health check (Streamlit built-in endpoint) ────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:${PORT}/_stcore/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD streamlit run app.py \
        --server.port=${PORT} \
        --server.address=0.0.0.0 \
        --server.headless=true \
        --server.enableCORS=false \
        --server.enableXsrfProtection=false \
        --browser.gatherUsageStats=false
