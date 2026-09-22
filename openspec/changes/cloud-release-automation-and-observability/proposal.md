## Why

The transaction model and cloud integrations are implemented locally, but a clean cloud deployment cannot reproduce the working service because model artifacts are git-ignored, CI still targets the legacy AWS path, and candidate mode retains a runtime dependency on legacy baseline artifacts. This change makes the transaction-serving path independently deployable on the selected free-tier providers and adds evidence-producing operational monitoring.

## What Changes

- Add a versioned model-artifact delivery contract using private Supabase Storage, integrity metadata, atomic local staging, and fail-fast readiness behavior.
- Replace the legacy AWS-oriented delivery workflow with CI checks and Render deployment automation for the selected cloud architecture.
- Make the transaction model the primary production runtime and remove its startup, readiness, and predeploy dependency on legacy baseline artifacts or the legacy CSV.
- Preserve historical baseline metrics as documentation/evidence while removing the legacy dataset from active registry, training, DVC, serving, and deployment paths.
- Add executable Evidently monitoring for prediction drift and delayed-label performance, with reports tied to model and reference-data versions.
- Add deployment smoke tests covering authentication, readiness, prediction, persistence, distributed rate limiting, and safe rollback behavior.
- Document Supabase migrations/storage setup, Upstash configuration, Render secrets, artifact promotion, rollback, and free-tier limitations.
- **BREAKING**: production configuration will default to transaction-model serving; the legacy manual baseline endpoint and legacy dataset pipeline will no longer be required for production startup.

## Capabilities

### New Capabilities

- `model-artifact-delivery`: Securely publish, download, verify, stage, and roll back versioned model artifact bundles through private Supabase Storage.
- `cloud-release-automation`: Run reproducible CI gates and deploy the verified transaction-serving release to Render with post-deployment smoke checks.

### Modified Capabilities

- `cloud-service-runtime`: Production startup and readiness will bootstrap the selected transaction artifact release without requiring baseline artifacts.
- `transaction-serving-promotion`: Transaction serving becomes the primary production path and is isolated from the legacy baseline model and dataset.
- `fraud-benchmark-datasets`: Historical baseline evidence remains documented, but the legacy CSV is removed from active dataset registry and executable pipeline dependencies.
- `fraud-observability`: Evidently drift and delayed-label reports become executable, versioned operational workflows rather than documentation-only support.
- `api-security-and-deploy-hardening`: Deployment preflight and smoke verification expand to cover remote artifact integrity, provider configuration, and rollback readiness.

## Impact

- Affected runtime and configuration: `app.py`, cloud settings, readiness, artifact loading, `render.yaml`, and provider predeploy checks.
- Affected ML/data paths: dataset registry, legacy ingestion/DVC configuration, candidate packaging metadata, and tests that currently assert legacy artifact behavior.
- Affected delivery: GitHub Actions, Docker build/runtime behavior, Supabase Storage and migrations, Upstash configuration, and Render deployment hooks.
- Affected monitoring: prediction/feedback extraction, Evidently report generation, report metadata/storage, and operational documentation.
- No raw transaction data or provider secrets will be committed. Historical benchmark results remain available for comparison, but production claims will be based on the transaction model and verified deployment evidence.
