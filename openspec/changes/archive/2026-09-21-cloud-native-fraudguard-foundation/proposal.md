## Why

FraudGuard is now technically defensible as an ML project, but it still runs like a single-machine demo: local artifacts, in-memory rate limits, no durable prediction history, no cloud deployment contract, and no first-class benchmark dataset workflow. This change prepares the project for a free-tier cloud-native deployment while preserving honest limits and a clear enterprise upgrade path.

## What Changes

- Add a cloud runtime contract for Render-compatible deployment, environment-driven configuration, startup artifact loading, health/readiness checks, and safe API behavior.
- Add Supabase-backed persistence for prediction requests, user feedback, model release metadata, and audit events.
- Add Upstash Redis support for distributed rate limiting and safe local fallback behavior.
- Add a benchmark dataset workflow that keeps the current dataset as the baseline and allows the user-downloaded benchmark dataset to be evaluated from separate training, test, and target artifacts without committing raw data.
- Change fraud model operating-policy requirements from F1-only thresholding toward explicit cost-weighted loss and cost-aware reporting.
- Add monitoring requirements for application telemetry and ML drift/performance reports using free-friendly tools such as Grafana Cloud and Evidently OSS.
- Add deploy configuration and documentation for Render, Supabase, and Upstash without provisioning external resources or embedding secrets in the repo.

## Capabilities

### New Capabilities

- `cloud-service-runtime`: FastAPI service behavior required for cloud deployment, environment configuration, artifact loading, health/readiness, and deploy config.
- `cloud-persistence-and-audit`: Supabase schema and application behavior for durable predictions, feedback, model releases, and audit trails.
- `distributed-rate-limiting`: Upstash-backed rate limiting behavior with explicit fallback and failure handling.
- `fraud-benchmark-datasets`: Dataset registry and benchmark workflow that preserves the current baseline and evaluates a separately downloaded benchmark dataset.
- `fraud-observability`: Application and ML monitoring behavior for logs, metrics, drift reports, delayed-label performance, and dashboard-ready outputs.

### Modified Capabilities

- `fraud-experiment-integrity`: Model selection, threshold selection, and final evaluation must support cost-weighted loss in addition to imbalance-aware ranking metrics.

## Impact

- Affected application areas: `app.py`, API response behavior, health/readiness endpoints, rate limiting, settings/configuration, logging, tests, and templates only where needed.
- Affected ML areas: training configuration, threshold selection metadata, evaluation metrics, dataset configuration, benchmark documentation, and artifact metadata.
- Affected deployment areas: `Dockerfile`, Render deployment config, environment variable docs, Supabase SQL migrations, Upstash setup docs, and CI checks.
- External systems planned but not provisioned by this change: Supabase, Upstash Redis, Render, Grafana Cloud, and Evidently OSS.
- Security impact: secrets must be read only from environment variables or provider secret stores; service-role database keys must never be sent to the browser or committed.
