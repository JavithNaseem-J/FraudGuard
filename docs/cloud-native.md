# Cloud-Native FraudGuard

FraudGuard is configured for a production-shaped free-tier deployment using Render, Supabase, Upstash Redis, and Evidently OSS. Free tiers are suitable for portfolio validation and pilot demos, not enterprise uptime commitments.

## Services

- Render: container web service for the FastAPI API.
- Supabase Postgres: prediction records, feedback, model releases, monitoring runs, and audit events.
- Supabase Storage: private immutable transaction model release bundles.
- Upstash Redis: distributed rate limiting for deployed API instances.
- Evidently OSS: bounded batch drift and delayed-label performance reports.

## Required Environment Variables

- `APP_ENV`: `local`, `render`, `staging`, or `production`.
- `FRAUD_MODEL_MODE`: `transaction_candidate` for production transaction serving. Use `baseline` only for explicit local historical demos.
- `TRANSACTION_ARTIFACT_RELEASE_ID`: immutable release ID selected for serving.
- `ARTIFACT_STORAGE_BUCKET`: private Supabase Storage bucket, for example `fraudguard-model-releases`.
- `ARTIFACT_CACHE_ROOT`: writable runtime cache. Render default: `/app/runtime/model-releases`.
- `TRANSACTION_CANDIDATE_ARTIFACT_ROOT`: local transaction artifact directory for offline validation.
- `SUPABASE_URL`: Supabase project URL.
- `SUPABASE_SERVICE_ROLE_KEY`: server-only service role key. Never expose it to browsers.
- `UPSTASH_REDIS_REST_URL`: Upstash Redis REST URL.
- `UPSTASH_REDIS_REST_TOKEN`: Upstash Redis REST token.
- `RATE_LIMIT_REQUESTS`: requests per window. Default: `20`.
- `RATE_LIMIT_WINDOW_SECONDS`: rate-limit window. Default: `60`.
- `UPSTASH_FAIL_CLOSED`: set to `true` to reject requests when Redis is unavailable.
- `AUTH_REQUIRED`: set to `true` for public deployments.
- `FRAUDGUARD_API_KEY`: server-side API key for protected prediction and feedback endpoints.
- `MAX_REQUEST_BYTES`: maximum JSON request body size. Default: `1048576`.
- `MAX_BATCH_ROWS`: maximum rows accepted by `/predict/transactions`. Default: `100`.

## Supabase Setup

1. Create a Supabase project.
2. Create a private Storage bucket for model releases.
3. Run the SQL files in `supabase/migrations/` in order.
4. Add `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `ARTIFACT_STORAGE_BUCKET`, and `TRANSACTION_ARTIFACT_RELEASE_ID` to Render environment variables.
5. Keep the service role key server-side only.

The migrations enable row-level security. The API service writes through server credentials; browser-facing public policies are intentionally not added.

## Artifact Promotion

Build or select a verified transaction model package containing:

- `model.joblib`
- `threshold.json`
- `metadata.json`
- `feature_audit.json`

Validate it locally:

```powershell
python scripts/publish_model_release.py --artifact-root artifacts/benchmark/transaction_data/candidate --release-id tx-YYYYMMDD-001 --local-only
```

Publish it to private Supabase Storage:

```powershell
python scripts/publish_model_release.py --artifact-root artifacts/benchmark/transaction_data/candidate --release-id tx-YYYYMMDD-001
```

Set `TRANSACTION_ARTIFACT_RELEASE_ID` to that immutable release ID. Rollback is the same operation with a prior verified release ID; release objects are never overwritten.

## Upstash Setup

1. Create an Upstash Redis database.
2. Copy the REST URL and REST token.
3. Add `UPSTASH_REDIS_REST_URL` and `UPSTASH_REDIS_REST_TOKEN` to Render.
4. Choose `UPSTASH_FAIL_CLOSED=true` for stricter public serving, or leave it false for demo availability during Redis outages.

## Render Setup

1. Connect this repository to Render.
2. Use `render.yaml` as the blueprint.
3. Confirm the health check path is `/live`.
4. Add provider secrets in the Render dashboard.
5. Set `AUTH_REQUIRED=true`, configure `FRAUDGUARD_API_KEY`, and select a published `TRANSACTION_ARTIFACT_RELEASE_ID`.
6. Run `python scripts/provider_predeploy.py` before deployment.
7. Deploy through the protected GitHub Actions Render hook after CI passes.

Render free services can sleep, cold-start, and throttle. Upgrade before promising uptime.

## API Security

Prediction and feedback endpoints are protected when `AUTH_REQUIRED=true`:

- `POST /predict/transactions`
- `POST /feedback`

The legacy `/predict` endpoint is disabled in production transaction mode. Health and readiness endpoints remain public so Render can monitor the service:

- `GET /live`
- `GET /ready`
- `GET /health`

Send the API key with either header:

```text
x-api-key: <FRAUDGUARD_API_KEY>
```

or:

```text
Authorization: Bearer <FRAUDGUARD_API_KEY>
```

Do not put API keys in URLs, templates, browser JavaScript, screenshots, or committed files.

## Monitoring

Use Evidently OSS for batch reports over sanitized prediction metadata and delayed labels. Reports should include release ID, model version, reference version, window, row counts, and generated time. They must not include raw transaction payloads or provider secrets.

Use `src/FraudGuard/monitoring/evidently_reports.py` for local file-based reports. Persist summary metadata to `monitoring_runs` when wiring scheduled jobs.

Langfuse is not part of this deployment because the system is tabular ML, not an LLM application.

## Dataset Benchmarking

The active benchmark registry expects user-downloaded transaction data:

- `data/train_transaction.csv`
- `data/train_identity.csv`
- `data/test_transaction.csv` when available
- `data/test_identity.csv` when available

There is no public test target file. Benchmark metrics must come from internal labeled splits of `train_transaction.csv`. Raw transaction data stays local and must not be committed.

## Free-Tier Limits

- Render services can sleep and cold-start.
- Supabase free projects have storage, compute, and retention limits.
- Upstash free Redis has request and storage quotas.
- Evidently OSS reports require a scheduler you control, such as GitHub Actions or a local/manual run.
- Local verification is not the same as deployed evidence. Record which checks ran locally and which ran against Render.
