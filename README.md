# FraudGuard

FraudGuard is a production-style transaction fraud scoring demonstration. It
shows how to train, validate, package, serve, monitor, and deploy an imbalanced
tabular classifier without claiming to be an enterprise banking platform.

The deployed shape is deliberately small:

```text
Browser -> React -> same-origin FastAPI -> verified model release
                                      |-> Supabase sanitized predictions
                                      |-> Upstash request counters
```

One Render Docker service serves both the compiled React application and the
FastAPI API. Raw transaction rows, uploaded files, card/customer/device
identifiers, and provider credentials are never persisted by the application.

## What is demonstrated

- Transaction-only model training from `train_transaction.csv`
- Deterministic chronological 70/15/15 train/validation/test periods
- Validation-only threshold selection with a documented 20:1 false-negative
  cost assumption and sensitivity at 10:1, 20:1, 50:1, and 100:1
- An untouched chronological test-period report with imbalance-aware metrics
- Promotion gates and immutable model bundles with SHA-256 checksums
- Anonymous public demo scoring bounded by 1 MiB requests, 100 rows per batch,
  and five requests per 60 seconds per client by default
- Server-only Supabase persistence and a 30-day dashboard snapshot
- Upstash distributed rate limiting with an explicit local-memory fallback
- Lightweight Evidently output monitoring using sanitized fields only
- Exact-commit GitHub Actions deployment to Render

## Deliberate limitations

This is not a bank-ready fraud platform. It does not implement authentication,
tenant isolation, investigator case management, payment blocking, streaming
ingestion, online feature computation, automatic retraining, regulatory
retention, disaster recovery, or paid uptime guarantees. Model scores are not
presented as calibrated fraud probabilities. The 20:1 cost ratio is a demo
assumption, not verified bank economics.

Render free services can cold-start. The UI displays loading/retry states, but
the project makes no availability guarantee.

## Repository structure

```text
app.py                         FastAPI entry point and static frontend serving
frontend/                      React, TypeScript, Vite, and Tailwind application
src/FraudGuard/cloud/          Settings, artifacts, persistence, rate limiting
src/FraudGuard/data/           Transaction-only preparation and evaluation
src/FraudGuard/pipeline/       Verified transaction-model inference
src/FraudGuard/monitoring/     Sanitized Evidently output reports
scripts/                       Training, release, monitoring, cleanup, smoke tools
supabase/migrations/           Reproducible database schema source
tests/                         Backend, ML-integrity, and API regression tests
dvc.yaml                       Reproducible bounded ML pipeline
```

Raw `data/`, generated `artifacts/`, local Supabase state, environments,
logs, tool output, and secrets remain untracked.

## Local setup

Python 3.11 or 3.12 and Node 20 are supported.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.lock
python -m pip install -e . --no-deps

Copy-Item .env.example .env
cd frontend
npm ci
npm run build
cd ..
python app.py
```

Open <http://127.0.0.1:8000>. Without Supabase and Upstash credentials, the
service starts in explicit `local_noop` persistence and `local_memory`
rate-limit modes. A valid local bundle must exist at
`artifacts/benchmark/transaction_data/model`, or configure an immutable
remote release.

For frontend development:

```powershell
# Terminal 1
python app.py

# Terminal 2
cd frontend
npm run dev
```

Vite proxies `/api/*` to `VITE_API_URL`.

## Database setup

Apply all files in `supabase/migrations/` in numeric order. Migration
`004_simplified_demo_predictions.sql` narrows persistence to:

- prediction and request IDs;
- timestamp and transaction amount;
- score, threshold, and decision;
- calibration flag and latency;
- model version and immutable release ID.

The service-role key is backend-only. Never place it in `VITE_*` variables or
browser code. The dashboard reads Supabase through `GET /dashboard`; the
browser never queries Supabase directly.

Dashboard reads are limited to the newest 10,000 records from the preceding 30
days. The response reports when it is truncated. Refresh fetches a new server
snapshot. Double-click Clear only resets the current browser view.

Retention cleanup is best-effort after persisted scoring. The owner can also run:

```powershell
python -m scripts.cleanup_predictions --older-than-days 30 --confirm
# or, intentionally:
python -m scripts.cleanup_predictions --all --confirm
```

There is no public deletion endpoint.

## Training and evaluation

`test_transaction.csv` is unlabeled and is used only for schema checks,
samples, or inference. All quality metrics come from chronological partitions
of the labeled training file.

Run a bounded workstation evaluation:

```powershell
python -m scripts.transaction_data_contract
python -m scripts.transaction_benchmark --sample-rows 75000
```

The verified bounded run completed in about three minutes on the development
machine. Peak resident memory is not claimed because the current runner does
not reliably capture native-library allocations; use `--full` only when memory
and runtime resources permit.

Run the complete local dataset only when machine resources allow:

```powershell
python -m scripts.transaction_benchmark --full
```

Or reproduce the bounded stages with DVC:

```powershell
dvc repro
```

Training removes exact duplicates before splitting and never divides equal
`TransactionDT` groups across periods. Preprocessing and fitting use the
training period; threshold selection uses validation; final metrics use the
later untouched test period. A partition without both classes fails rather
than falling back to a random split.

Initial demonstration gates are:

- average precision >= 0.70;
- recall >= 0.70;
- average cost <= 0.20;
- average cost no worse than the chronological logistic baseline;
- valid feature schema and artifact package.

A failed evaluation remains useful evidence but cannot be published:

```powershell
python -m scripts.publish_model_release `
  --artifact-root artifacts/benchmark/transaction_data/evaluated-model `
  --release-id tx-YYYYMMDD-001 `
  --local-only
```

Remove `--local-only` only after configuring the private Supabase Storage
bucket and server credentials.

## Monitoring

Evidently is a development dependency, not a production runtime dependency.
It compares sanitized reference and current exports containing only timestamp,
amount, score, threshold, decision, latency, model version, and release ID.

```powershell
python -m scripts.generate_monitoring_report `
  --reference reference_predictions.csv `
  --current current_predictions.csv `
  --release-id tx-YYYYMMDD-001
```

The command emits JSON status plus an HTML report when both windows have enough
rows. Empty or small windows produce explicit `no_data` or
`insufficient_data` status. Delayed-label monitoring is intentionally absent
because this demo has no trusted reviewer-label workflow.

## Verification

```powershell
black --check app.py src tests scripts
flake8 app.py src tests scripts --max-line-length=120 --extend-ignore=E203,W503
python -m mypy app.py src/FraudGuard/cloud src/FraudGuard/pipeline
python -m pytest -p no:cacheprovider

cd frontend
npm test
npm audit --omit=dev --audit-level=moderate
npm run build
cd ..

docker build -t fraudguard:local .
```

## Render deployment

Create one Render Blueprint service from `render.yaml`. Do not recreate an
existing live service. In the Render dashboard, set:

- `TRANSACTION_ARTIFACT_RELEASE_ID`
- `ARTIFACT_STORAGE_BUCKET`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `UPSTASH_REDIS_REST_URL`
- `UPSTASH_REDIS_REST_TOKEN`

`autoDeploy: false` is intentional. Configure these GitHub repository or
environment secrets:

- `RENDER_DEPLOY_HOOK_URL`
- `RENDER_PUBLIC_BASE_URL`

After a push to `main`, CI verifies backend, frontend, and the production
container. Only a successful CI run can trigger deployment. The deployment
workflow passes the exact tested commit SHA to Render, then verifies
`/version`, `/ready`, anonymous scoring, and `/dashboard`.

To roll back, set `TRANSACTION_ARTIFACT_RELEASE_ID` to a previously verified
immutable release and deploy the corresponding previously tested Git commit.
The same manifest, checksum, schema, loading, and readiness checks apply.

## Interview defense

The design favors demonstrable controls over unused enterprise scaffolding:

- chronological evaluation is less flattering than random splitting but better
  represents future transaction scoring;
- a logistic baseline establishes whether LightGBM earns its complexity;
- validation-only threshold selection protects the final test period;
- cost sensitivity exposes dependence on an assumed business ratio;
- anonymous access is acceptable only because payload, batch, and distributed
  rate limits bound this public demo;
- sanitized persistence supports a credible dashboard without retaining the
  full transaction feature payload;
- one container avoids CORS and two-service deployment state;
- immutable releases and exact-commit delivery demonstrate rollback and
  traceability without adding an automated model-control plane.

Only claim metrics from an executed chronological report. Do not reuse older
random-split headline scores as production evidence.
