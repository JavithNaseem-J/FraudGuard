# FraudGuard

Transaction fraud scoring with chronological ML evaluation, cost-aware decisions,
sanitized monitoring, and exact-commit deployment.

**Production URL:** <https://fraudguard.onrender.com>

## Problem

Fraud datasets are highly imbalanced, and a high accuracy score can hide a model
that misses most fraudulent transactions. FraudGuard demonstrates a more useful
workflow: preserve time order, tune the decision threshold on validation data,
measure the result on a later untouched period, and treat missed fraud as more
expensive than a false alert.

This repository is a production-style portfolio project, not a bank-ready payment
system. It shows the engineering around a fraud model without claiming enterprise
availability, regulatory compliance, or automated transaction blocking.

## Features

- **Transaction-only training:** trains from labeled transaction data; identity
  side tables are deliberately excluded from the current benchmark.
- **Time-aware evaluation:** uses deterministic 70/15/15 chronological
  train/validation/test periods and keeps equal transaction-time groups together.
- **Cost-aware threshold:** selects the threshold on validation data with a 20:1
  false-negative-to-false-positive cost assumption and reports sensitivity at
  10:1, 20:1, 50:1, and 100:1.
- **Strong and baseline models:** compares a LightGBM classifier with a
  chronological logistic-regression baseline.
- **Immutable releases:** packages model, threshold, metadata, and feature audit
  files behind a SHA-256 manifest before publication to private Supabase Storage.
- **Bounded public scoring:** accepts JSON or CSV through the React interface,
  with a 1 MiB request limit, 100-row batch limit, and a default rate limit of
  five requests per 60 seconds per client.
- **Sanitized dashboard:** stores prediction metadata in Supabase without storing
  raw transaction rows or uploaded files, then serves a bounded 30-day snapshot.
- **Deployment traceability:** GitHub Actions tests the production image, deploys
  the exact passing commit to Render, and verifies the live `/version` response.

## System architecture

```text
Browser
  |
  v
React + TypeScript dashboard
  |
  | same-origin HTTP
  v
FastAPI service
  |-- verified transaction model release
  |-- Supabase Postgres: sanitized prediction records
  |-- Supabase Storage: immutable model bundles
  `-- Upstash Redis: distributed request counters

GitHub Actions -- exact tested commit --> Render Docker service
```

One Docker container serves both the compiled frontend and the API. This keeps
the demo deployable on one free-tier Render service and avoids a second frontend
service and cross-origin configuration.

## Prediction lifecycle

1. The browser loads the active feature schema from `GET /schema/transactions`.
2. A user pastes rows or uploads a CSV file; parsing and preview happen in the
   browser.
3. The frontend sends at most 100 normalized rows to
   `POST /predict/transactions`.
4. FastAPI enforces request size and rate limits, validates the exact feature
   contract, and scores each row with the loaded release.
5. The model returns a fraud score, configured threshold, and `Yes`/`No` flag.
   The score is not presented as a calibrated fraud probability.
6. The backend persists only sanitized prediction metadata when Supabase is
   configured. The dashboard reads that server-side history from `GET /dashboard`.

Raw transaction rows, card/customer/device identifiers, uploaded files, and
provider credentials are not persisted by the application.

## API endpoints

The interactive OpenAPI documentation is available at `/docs`.

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/live` | Lightweight process liveness check. |
| `GET` | `/health` | Alias for the liveness check. |
| `GET` | `/ready` | Confirms that the transaction model is loaded and reports dependency modes. |
| `GET` | `/version` | Returns only the deployed commit SHA and UTC build time. |
| `GET` | `/schema/transactions` | Returns the active model schema, threshold, and batch limit. |
| `POST` | `/predict/transactions` | Scores a JSON batch of transaction rows. |
| `GET` | `/dashboard` | Returns sanitized prediction aggregates and recent flagged records. |
| `GET` | `/api` | Returns the service index and endpoint links. |
| `GET` | `/` | Serves the compiled React application. |
| `GET` | `/score` | Serves the scoring view for client-side routing. |
| `GET` | `/sample_transactions.csv` | Serves the frontend sample transaction file. |

The public demo is anonymous and rate-limited. There is no browser API key and
no public endpoint that deletes persisted predictions.

## Tech stack

| Layer | Technology |
|---|---|
| Model | LightGBM 4.7, scikit-learn 1.8, pandas, NumPy |
| API | FastAPI 0.141, Uvicorn, Pydantic 2 |
| Frontend | React 18, TypeScript, Vite, Tailwind CSS, Recharts |
| Persistence | Supabase Postgres through server-only REST calls |
| Model storage | Private Supabase Storage with immutable release manifests |
| Rate limiting | Upstash Redis REST with local-memory fallback |
| Monitoring | Evidently reports over sanitized prediction exports |
| Reproducibility | Locked Python/npm dependencies and DVC stages |
| Deployment | Docker, Render, GitHub Actions |
| Runtime | Python 3.11 or 3.12; Node 20 for frontend builds |

## Evaluation evidence

The latest bounded local benchmark used 75,000 labeled rows and evaluated the
final model on a later 15,000-row chronological test period. These numbers are
development evidence, not a claim about live banking performance.

| Metric | Result | Interpretation |
|---|---:|---|
| Average precision | `0.7090` | More informative than accuracy for this imbalanced target. |
| Recall | `0.8218` | Detected 332 of 404 positive rows in the held-out period. |
| Precision | `0.2265` | The cost-aware threshold intentionally accepts more review alerts. |
| F1 | `0.3551` | Balance of precision and recall at the selected threshold. |
| ROC AUC | `0.9401` | Ranking quality across thresholds. |
| Selected threshold | `0.2172` | Chosen only from the chronological validation period. |
| Average weighted cost | `0.1716` | Uses false-positive cost 1 and false-negative cost 20. |
| Test prevalence | `2.69%` | 404 positive rows among 15,000 test rows. |

The unlabeled test transaction file is used only for schema checks, sample rows,
and inference. It is never used to calculate model quality. Exact duplicates are
removed before splitting, preprocessing is fitted only on training data, and a
partition without both classes fails rather than silently switching to a random
split.

Initial demonstration gates are average precision >= 0.70, recall >= 0.70,
average cost <= 0.20, cost no worse than the chronological logistic baseline,
and a valid feature/artifact package. Only reports produced by the chronological
pipeline should be cited as model evidence.

## Local setup

### 1. Clone and create the Python environment

```powershell
git clone https://github.com/JavithNaseem-J/FraudGuard.git
cd FraudGuard

python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install -e . --no-deps
```

### 2. Configure local environment variables

```powershell
Copy-Item .env.example .env
```

For local scoring, the default artifact path is
`artifacts/benchmark/transaction_data/model`. Supabase and Upstash credentials
are optional locally; without them the service reports `local_noop` persistence
and `local_memory` rate limiting.

### 3. Build the frontend

```powershell
cd frontend
npm ci
npm run build
cd ..
```

### 4. Start the application

```powershell
python app.py
```

Open <http://127.0.0.1:8000>. For frontend development, keep the API running in
one terminal and run `npm run dev` from `frontend/` in another. Vite serves
<http://127.0.0.1:5173> and proxies `/api/*` to the configured backend.

## Database and model storage

Apply the SQL files in `supabase/migrations/` in numeric order. The latest schema
stores only prediction/request IDs, timestamp, amount, score, threshold,
decision, calibration flag, latency, model version, and immutable release ID.

The following values are server-only and must never be placed in `VITE_*`
variables or frontend code:

```dotenv
SUPABASE_URL=
SUPABASE_SERVICE_ROLE_KEY=
ARTIFACT_STORAGE_BUCKET=fraudguard-model-releases
TRANSACTION_ARTIFACT_RELEASE_ID=
UPSTASH_REDIS_REST_URL=
UPSTASH_REDIS_REST_TOKEN=
```

The dashboard reads Supabase through the FastAPI service. It returns at most the
newest 10,000 predictions from the preceding 30 days and reports when that
window is truncated. Refresh requests a new server snapshot. Double-clicking
Clear resets only the current browser view.

## Training and release

Run the data contract and bounded chronological benchmark:

```powershell
python -m scripts.transaction_data_contract
python -m scripts.transaction_benchmark --sample-rows 75000
```

Use all labeled rows only when the machine has enough memory and runtime:

```powershell
python -m scripts.transaction_benchmark --full
```

The bounded DVC stages can also be reproduced with:

```powershell
dvc repro
```

Publication is blocked unless every promotion gate passes:

```powershell
python -m scripts.publish_model_release `
  --artifact-root artifacts/benchmark/transaction_data/evaluated-model `
  --release-id tx-YYYYMMDD-001 `
  --local-only
```

Remove `--local-only` only after configuring the private Supabase Storage bucket
and server credentials. A published release contains `manifest.json`,
`model.joblib`, `threshold.json`, `metadata.json`, and `feature_audit.json`.

## Monitoring

Evidently is a development dependency rather than a production runtime service.
It compares sanitized reference and current exports containing only timestamp,
amount, score, threshold, decision, latency, model version, and release ID.

```powershell
python -m scripts.generate_monitoring_report `
  --reference reference_predictions.csv `
  --current current_predictions.csv `
  --release-id tx-YYYYMMDD-001
```

The command emits a machine-readable status and an HTML report when both windows
contain enough rows. Empty or small windows produce explicit `no_data` or
`insufficient_data` status.

## Verification

```powershell
black --check app.py src tests scripts
flake8 app.py src tests scripts --max-line-length=120 --extend-ignore=E203,W503
python -m mypy app.py src/FraudGuard/cloud src/FraudGuard/pipeline
python -m pytest -p no:cacheprovider

cd frontend
npm run typecheck
npm test
npm audit --omit=dev --audit-level=moderate
npm run build
cd ..
```

## Docker

```powershell
docker build -t fraudguard:local .
docker run --env-file .env -p 8000:8000 fraudguard:local
```

The multi-stage image compiles the React frontend with Node 20, installs the
locked Python runtime, runs as a non-root user, serves both application layers
with Uvicorn, binds to `${PORT:-8000}`, and checks `/ready` for container health.

## Deployment

`render.yaml` defines one free-tier Docker service named `fraudguard` with
`autoDeploy: false`. GitHub Actions, not Render's direct Git integration, decides
when a commit is safe to deploy.

The deployment flow is:

1. `.github/workflows/ci.yml` runs on pushes and pull requests to `main`, plus
   manual dispatch.
2. Backend formatting, linting, type analysis, unit tests, and integration tests
   must pass.
3. Frontend lockfile installation, typecheck, tests, dependency audit, and
   production build must pass.
4. CI builds the production Docker image with the commit SHA and UTC build time,
   starts it, and verifies readiness, version identity, scoring, dashboard, and
   frontend serving.
5. `.github/workflows/render-deploy.yml` runs only after successful CI on `main`.
6. The deploy hook receives `ref=<exact-commit-sha>`. GitHub waits for Render and
   fails unless the live `/version` reports that same SHA.

Configure these secrets on the GitHub `production` environment:

```text
RENDER_DEPLOY_HOOK_URL
RENDER_PUBLIC_BASE_URL=https://fraudguard.onrender.com
```

In the Render dashboard, keep the existing service rather than recreating it.
Confirm its name and URL, disable direct Git auto-deploy and Blueprint Auto Sync,
keep the deploy hook for that service, and provide the server-only values listed
in the database section. Existing approval protection on the GitHub `production`
environment should remain enabled.

## Project structure

```text
FraudGuard/
|-- .github/workflows/       CI and exact-commit Render deployment
|-- frontend/                React, TypeScript, Vite, and Tailwind application
|-- scripts/                 Training, release, monitoring, cleanup, and smoke tools
|-- src/FraudGuard/
|   |-- cloud/               Settings, artifacts, persistence, and rate limiting
|   |-- data/                Transaction preparation and chronological evaluation
|   |-- monitoring/          Sanitized Evidently report generation
|   `-- pipeline/            Model artifact validation and inference
|-- supabase/migrations/     Reproducible database schema
|-- tests/                   Backend, ML-integrity, and API regression tests
|-- app.py                   FastAPI entry point and frontend serving
|-- Dockerfile               Production multi-stage container
|-- dvc.yaml                 Reproducible bounded ML stages
|-- pyproject.toml           Python package and tool configuration
`-- render.yaml              Single-service Render blueprint
```

Raw data, generated artifacts, local Supabase state, virtual environments, logs,
tool output, and secrets are intentionally untracked.

## Deliberate limitations and future work

- No authentication, tenant isolation, investigator workflow, or transaction
  blocking. Anonymous access is acceptable only for this bounded demo.
- No streaming ingestion or online feature store; scoring is request/response
  batch inference.
- No automated retraining or unattended model promotion. Release publication is
  an explicit owner action after evaluation.
- No reviewer labels or delayed-label performance monitoring. Current monitoring
  covers sanitized prediction-output drift only.
- No multi-region failover, disaster-recovery target, regulatory retention
  policy, or paid uptime guarantee. Render free services can cold-start.
- The 20:1 cost ratio is a documented demonstration assumption, not verified
  banking economics.

## License

This project is licensed under the [MIT License](LICENSE).
