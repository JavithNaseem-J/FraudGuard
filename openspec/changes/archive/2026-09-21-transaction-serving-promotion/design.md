## Context

The latest transaction-data candidate passed local evidence gates on a bounded 75,000-row run. Its feature schema contains hundreds of benchmark transaction features, while the current web/API prediction path accepts a compact manual fraud form. Directly swapping the model into the existing endpoint would be unsafe because the served input schema would not match training.

## Design Summary

Use a feature-flagged serving promotion path:

1. Keep the existing `/predict` behavior as the default serving path.
2. Add a transaction-candidate loader that reads the candidate `model.joblib`, `threshold.json`, `metadata.json`, and `feature_audit.json`.
3. Add a batch transaction prediction contract for rows that already match the benchmark feature schema.
4. Gate the candidate path behind environment configuration such as `FRAUD_MODEL_MODE=baseline|transaction_candidate`.
5. Return per-row prediction IDs, scores, thresholds, decisions, model version/metadata, and schema-validation errors.
6. Persist prediction and model-release metadata to Supabase when configured; otherwise use the existing local fallback.
7. Reuse Upstash rate limiting for the new endpoint when configured.
8. Emit structured telemetry without logging raw payloads or secrets.

## Approach Options Considered

### Option A: Replace existing `/predict`

Rejected for now. It is simpler to demo but unsafe because the existing request schema does not contain the candidate's required transaction features.

### Option B: Add `/predict/transactions` batch endpoint first

Recommended. This matches the candidate feature contract, avoids training-serving skew, supports CSV/JSON row-style prediction later, and keeps the current user-facing endpoint stable.

### Option C: Build a feature store first

Deferred. A feature store would be closer to enterprise production, but it is too large for the next change and would slow down getting a safe candidate-serving milestone.

## Components

- **Candidate artifact loader:** validates model, threshold, metadata, feature list, and schema audit before serving.
- **Transaction batch predictor:** accepts one or more transaction rows with the exact required features, orders columns according to metadata, and produces scores/decisions.
- **Runtime mode selector:** chooses baseline or transaction candidate from environment variables.
- **API endpoint:** exposes batch prediction without changing the current manual `/predict` endpoint.
- **Persistence adapter:** stores prediction metadata and optional feedback links in Supabase.
- **Telemetry hooks:** records latency, model mode, model version, row count, status, and sanitized error category.

## Error Handling

- Missing candidate artifacts fail readiness for candidate mode.
- Missing required transaction features return a validation error naming missing fields.
- Extra features may be ignored only if the response records that they were ignored.
- Provider outages must not expose secrets or raw payloads in errors.
- Baseline mode must continue to work when candidate artifacts are absent.

## Testing Plan

- Unit test candidate artifact validation.
- Unit test feature ordering, missing-feature errors, and extra-feature behavior.
- API test batch prediction success and validation failure.
- Readiness test baseline mode vs candidate mode.
- Persistence/rate-limit tests using local fallbacks.
- Render predeploy check for required environment variables.

## Non-Goals

- No full feature store in this change.
- No browser UI rebuild in this change.
- No paid provider dependency.
- No claim that public unlabeled test files provide final metrics.
- No destructive raw data edits.
