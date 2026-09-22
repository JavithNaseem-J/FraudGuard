## Why

The transaction-data benchmark candidate now has strong internal evidence, but it cannot be safely served until the API can load the candidate artifact, validate the wider transaction feature schema, preserve the current serving fallback, and expose production-grade audit/observability metadata.

## What Changes

- Add a feature-flagged transaction candidate serving path.
- Add a batch-style transaction prediction contract that accepts transaction benchmark rows and returns per-row decisions.
- Keep the existing manual prediction endpoint available as the default fallback until the candidate is explicitly enabled.
- Add readiness, audit, and observability requirements for promoted transaction candidates.
- Document deployment expectations for Render/Supabase/Upstash without committing secrets.

## Capabilities

### New Capabilities

- `transaction-serving-promotion`: Covers feature-flagged loading, schema validation, batch prediction, fallback behavior, and deployment checks for transaction-data model serving.

### Modified Capabilities

- `fraud-inference-contract`: Add candidate artifact compatibility requirements for transaction-data serving.
- `cloud-service-runtime`: Add candidate-aware readiness and safe runtime selection.
- `cloud-persistence-and-audit`: Add prediction/model-release audit requirements for transaction candidate serving.
- `fraud-observability`: Add transaction candidate telemetry and drift/reporting expectations.
- `transaction-model-promotion`: Clarify that a candidate may be promoted only through an explicit serving-promotion flow.

## Impact

- Affected API: likely adds a new batch transaction prediction endpoint while preserving current `/predict`.
- Affected model loading: candidate artifact directory under `artifacts/benchmark/transaction_data/candidate/`.
- Affected cloud config: feature flag and artifact path environment variables for Render/local use.
- Affected persistence/observability: Supabase prediction/model-release records, Upstash rate limiting, structured telemetry.
- No raw dataset changes.
- No automatic replacement of the current serving model.
