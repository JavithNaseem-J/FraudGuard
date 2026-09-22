## Context

The service can now run locally, expose readiness/liveness checks, persist prediction metadata to Supabase, reuse Upstash/local rate limiting, and serve the transaction candidate behind a feature flag. The remaining gap is public-deployment hardening: a Render URL without authentication or request limits would invite accidental abuse, noisy logs, and avoidable provider cost/rate-limit issues.

## Design Summary

Implement a small but useful production-demo hardening layer:

1. Add optional API-key authentication controlled by environment variables.
2. Protect state-changing and prediction endpoints while keeping `/live`, `/ready`, `/health`, and static/public pages available.
3. Add request body and batch row limits with explicit 413/422-style failures.
4. Extend the predeploy check to validate cloud settings, candidate artifacts when enabled, and likely missing secrets without printing secret values.
5. Document artifact delivery options and recommend an object-storage or deploy-time artifact-copy path for real deployments.
6. Update README/docs with final local/cloud commands, verified metrics, and limitations.

## Approach Options Considered

### Option A: Simple server API key

Recommended for this change. It is enough for a portfolio/demo API, works on Render free tier, and avoids introducing a database-backed user system before the product needs one.

### Option B: Full user auth with JWT/session accounts

Deferred. Better for a multi-user SaaS product, but too much scope for the next milestone because it requires identity, signup/login, roles, key rotation, and UI flows.

### Option C: Leave endpoints public and rely on rate limits

Rejected. Rate limiting reduces abuse but does not establish caller authorization, and public prediction endpoints can still leak capacity or create noisy data.

## Configuration

Proposed environment variables:

- `FRAUDGUARD_API_KEY`: server-side key required for protected endpoints when set.
- `AUTH_REQUIRED`: when true, reject protected requests that omit or mismatch the API key.
- `MAX_REQUEST_BYTES`: maximum request body size for JSON requests.
- `MAX_BATCH_ROWS`: maximum rows accepted by `/predict/transactions`.

Local defaults should be permissive enough for development, while Render docs should recommend enabling auth before making the service public.

## Protected Endpoint Policy

Protected:

- `POST /predict`
- `POST /predict/transactions`
- `POST /feedback`

Public:

- `GET /live`
- `GET /ready`
- `GET /health`
- `GET /`
- static assets

Accepted API-key locations:

- `x-api-key` header
- `Authorization: Bearer <key>`

## Error Handling

- Missing API key -> `401 Unauthorized`
- Wrong API key -> `403 Forbidden`
- Oversized body -> `413 Payload Too Large`
- Oversized batch -> validation error with configured limit
- Predeploy failures -> nonzero exit code with sanitized diagnostics
- No provider secrets in errors, logs, docs examples, or rendered pages

## Testing Plan

- Unit/API tests for missing, wrong, and valid API keys.
- Tests that health/readiness remain public.
- Tests for request body size and batch row limits.
- Predeploy script tests or command smoke checks for baseline and candidate modes.
- Compile checks and OpenSpec validation.

## Non-Goals

- No full identity provider or role-based access control.
- No payment/billing.
- No automatic upload of artifacts to third-party object storage.
- No destructive cleanup of local artifacts or datasets.
