# FraudGuard

FraudGuard is a tabular transaction-fraud project with a FastAPI serving API, transaction-data benchmark workflow, immutable model release handling, and free-tier cloud deployment configuration for Render, Supabase, Upstash Redis, and Evidently OSS.

Production serving now uses the transaction model path. The older single-CSV baseline is retained only as historical comparison evidence and is not required for production startup, CI, DVC, Render, or predeploy checks.

## Production Path

1. Train or select a transaction model package from local transaction data.
2. Validate the package contract and write a manifest with file sizes and SHA-256 checksums.
3. Publish the immutable release to private Supabase Storage.
4. Configure Render with `FRAUD_MODEL_MODE=transaction_candidate` and `TRANSACTION_ARTIFACT_RELEASE_ID`.
5. On startup, the API downloads or reuses the verified cached release, validates integrity, loads the model, and exposes readiness only after the selected release is usable.
6. Authenticated `/predict/transactions` requests are rate-limited through Upstash and can persist sanitized prediction metadata to Supabase.
7. Evidently batch jobs can produce drift and delayed-label performance reports from sanitized records.

## Current Transaction Model Evidence

The latest bounded transaction-data candidate run used 75,000 labeled rows and evaluated on a 15,000-row internal labeled split:

| Metric | Value |
| --- | ---: |
| Average precision / PR-AUC | 0.7192 |
| ROC-AUC | 0.9409 |
| Precision at threshold | 0.3033 |
| Recall at threshold | 0.7921 |
| F1 at threshold | 0.4387 |
| Brier score | 0.0209 |
| Cost-weighted average loss | 0.1610 |

There is no public test target in this workspace, so public test performance is not claimed.

## Local Setup

Use Python 3.9-3.11 for the locked dependency set.

```bash
python -m pip install -r requirements.lock
python -m pip install -e .
pytest -q -p no:cacheprovider
```

Validate the active transaction data contract:

```bash
python scripts/transaction_data_contract.py
```

Validate a local transaction model package:

```bash
python scripts/publish_model_release.py --artifact-root artifacts/benchmark/transaction_data/candidate --release-id local-check --local-only
```

Run deployment preflight:

```bash
python scripts/provider_predeploy.py
```

## Cloud Deployment

Render reads `render.yaml`. Required protected values include:

- `FRAUDGUARD_API_KEY`
- `SUPABASE_URL`
- `SUPABASE_SERVICE_ROLE_KEY`
- `ARTIFACT_STORAGE_BUCKET`
- `TRANSACTION_ARTIFACT_RELEASE_ID`
- `UPSTASH_REDIS_REST_URL`
- `UPSTASH_REDIS_REST_TOKEN`

GitHub Actions runs clean-checkout tests, compile checks, predeploy validation, container build, and a protected Render deploy hook.

## API

Health endpoints:

- `GET /live`
- `GET /ready`
- `GET /health`

Production prediction endpoint:

- `POST /predict/transactions`

Protected endpoints require either:

```text
x-api-key: <FRAUDGUARD_API_KEY>
```

or:

```text
Authorization: Bearer <FRAUDGUARD_API_KEY>
```

The legacy `/predict` endpoint is available only when `FRAUD_MODEL_MODE=baseline` is explicitly selected for local historical demos.

## Dataset Notes

The active transaction workflow expects local files such as:

- `data/train_transaction.csv`
- `data/train_identity.csv`
- `data/test_transaction.csv`
- `data/test_identity.csv`

Raw transaction data and model binaries are intentionally not committed. The active registry does not require the retired legacy CSV.

## Limits

- Scores are model scores, not calibrated financial risk.
- The current evidence uses internal labeled splits, not public test labels.
- Free-tier Render can sleep and cold-start.
- API-key auth is sufficient for a protected demo, not multi-user enterprise identity.
- A real enterprise rollout still needs key rotation, operator workflows, alerting, backup policy, access reviews, and paid uptime choices.
