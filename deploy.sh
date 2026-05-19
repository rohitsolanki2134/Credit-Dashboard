#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# deploy.sh — One-time setup + manual deploy for Cloud Run
# ─────────────────────────────────────────────────────────────────────────────
# Usage:
#   chmod +x deploy.sh
#   ./deploy.sh                   # full setup + deploy
#   ./deploy.sh --deploy-only     # skip setup, just build & deploy
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── CONFIG — edit these before running ────────────────────────────────────────
PROJECT_ID="your-gcp-project-id"        # ← REPLACE with your GCP project ID
REGION="asia-south1"                     # Mumbai; change if needed
SERVICE="credit-dashboard"
IMAGE="gcr.io/${PROJECT_ID}/${SERVICE}"

# Secrets — loaded from your local .env automatically when you run:
#   source .env && ./deploy.sh
# Or hard-code them here (keep this file out of git if you do that):
ANTHROPIC_API_KEY_VAL="${ANTHROPIC_API_KEY:-}"
SMTP_USER_VAL="${SMTP_USER:-btcinvoice@travelplusapp.com}"
SMTP_PASSWORD_VAL="${SMTP_PASSWORD:-vdmuqcpnzraeabws}"
# ──────────────────────────────────────────────────────────────────────────────

DEPLOY_ONLY=false
for arg in "$@"; do
  [[ "$arg" == "--deploy-only" ]] && DEPLOY_ONLY=true
done

echo "🚀  TravelPlus Credit Dashboard — Cloud Run deploy"
echo "    Project : ${PROJECT_ID}"
echo "    Region  : ${REGION}"
echo "    Service : ${SERVICE}"
echo ""

# ── Authenticate & set project ────────────────────────────────────────────────
gcloud config set project "${PROJECT_ID}"

if [[ "$DEPLOY_ONLY" == false ]]; then
  echo "── Step 1: Enable required APIs ─────────────────────────────────────────"
  gcloud services enable \
    cloudbuild.googleapis.com \
    run.googleapis.com \
    secretmanager.googleapis.com \
    containerregistry.googleapis.com

  echo ""
  echo "── Step 2: Create secrets in Secret Manager ──────────────────────────────"
  echo "   (Skipped if secrets already exist)"

  create_secret() {
    local name=$1
    local value=$2
    if gcloud secrets describe "${name}" --project="${PROJECT_ID}" &>/dev/null; then
      echo "   Secret '${name}' already exists — adding a new version."
      echo -n "${value}" | gcloud secrets versions add "${name}" --data-file=-
    else
      echo "   Creating secret '${name}'."
      echo -n "${value}" | gcloud secrets create "${name}" \
        --data-file=- \
        --replication-policy=automatic
    fi
  }

  [[ -n "${ANTHROPIC_API_KEY_VAL}" ]] && create_secret "ANTHROPIC_API_KEY" "${ANTHROPIC_API_KEY_VAL}"
  [[ -n "${SMTP_USER_VAL}" ]]         && create_secret "SMTP_USER"         "${SMTP_USER_VAL}"
  [[ -n "${SMTP_PASSWORD_VAL}" ]]     && create_secret "SMTP_PASSWORD"     "${SMTP_PASSWORD_VAL}"

  echo ""
  echo "── Step 3: Grant Cloud Build IAM permissions ─────────────────────────────"
  PROJECT_NUMBER=$(gcloud projects describe "${PROJECT_ID}" --format='value(projectNumber)')
  CB_SA="${PROJECT_NUMBER}@cloudbuild.gserviceaccount.com"
  COMPUTE_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${CB_SA}" \
    --role="roles/run.admin" --quiet

  gcloud projects add-iam-policy-binding "${PROJECT_ID}" \
    --member="serviceAccount:${CB_SA}" \
    --role="roles/secretmanager.secretAccessor" --quiet

  gcloud iam service-accounts add-iam-policy-binding "${COMPUTE_SA}" \
    --member="serviceAccount:${CB_SA}" \
    --role="roles/iam.serviceAccountUser" --quiet

  echo ""
fi

echo "── Step 4: Build & deploy via Cloud Build ────────────────────────────────"
gcloud builds submit --config cloudbuild.yaml \
  --substitutions="_REGION=${REGION},_SERVICE=${SERVICE}"

echo ""
echo "✅  Deploy complete!"
echo "    Service URL: $(gcloud run services describe ${SERVICE} --region=${REGION} --format='value(status.url)')"
