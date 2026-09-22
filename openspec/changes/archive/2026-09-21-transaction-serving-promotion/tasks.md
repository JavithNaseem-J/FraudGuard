## 1. Serving Contract

- [x] 1.1 Add candidate artifact loading and validation for model, threshold, metadata, and feature audit.
- [x] 1.2 Add runtime mode selection for baseline vs transaction candidate.
- [x] 1.3 Add batch transaction prediction schema validation and feature ordering.
- [x] 1.4 Add candidate batch prediction endpoint while preserving current `/predict`.

## 2. Cloud Runtime and Deployment

- [x] 2.1 Update readiness to report baseline/candidate mode and sanitized artifact status.
- [x] 2.2 Add candidate-mode environment variables to Render/local deployment docs.
- [x] 2.3 Add or update predeploy checks for candidate artifact files and required runtime settings.

## 3. Persistence, Rate Limit, and Observability

- [x] 3.1 Persist candidate prediction metadata and model mode through Supabase when configured.
- [x] 3.2 Register candidate model release metadata without secrets or raw training data.
- [x] 3.3 Reuse Upstash/local rate limiting for the candidate endpoint.
- [x] 3.4 Emit structured telemetry for batch row count, latency, model mode, and sanitized validation errors.

## 4. Verification

- [x] 4.1 Add tests for candidate artifact validation and missing-artifact readiness behavior.
- [x] 4.2 Add tests for batch prediction success, missing features, feature ordering, and current endpoint fallback.
- [x] 4.3 Add tests for local Supabase fallback and rate-limit behavior on the candidate endpoint.
- [x] 4.4 Run unit tests, compile checks, OpenSpec validation, and an API smoke test.
