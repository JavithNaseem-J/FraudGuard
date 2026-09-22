## 1. API Security

- [x] 1.1 Add environment-driven API-key settings with safe local defaults.
- [x] 1.2 Protect `/predict`, `/predict/transactions`, and `/feedback`.
- [x] 1.3 Keep `/live`, `/ready`, `/health`, `/`, and static assets public.
- [x] 1.4 Add sanitized denied-request logging/audit metadata.

## 2. Request Guardrails

- [x] 2.1 Add configurable max request body size.
- [x] 2.2 Add configurable max batch rows for transaction candidate predictions.
- [x] 2.3 Ensure guardrails run before expensive prediction work.

## 3. Deployment Hardening

- [x] 3.1 Extend Render environment configuration for auth and request limits.
- [x] 3.2 Extend predeploy checks for auth posture, baseline artifacts, provider settings, and candidate artifacts when enabled.
- [x] 3.3 Document artifact delivery options for Render/local deployment.
- [x] 3.4 Confirm Supabase migration order and Upstash setup are documented.

## 4. Documentation and Claims

- [x] 4.1 Update README with architecture, local run, candidate mode, deployment, metrics, and limitations.
- [x] 4.2 Keep professional "transaction data" wording in docs and user-facing outputs.
- [x] 4.3 Add safe resume/portfolio claims and explicitly list what is not yet enterprise-grade.

## 5. Verification

- [x] 5.1 Add tests for missing, invalid, and valid API keys.
- [x] 5.2 Add tests for public health/readiness and protected prediction endpoints.
- [x] 5.3 Add tests for request-size and batch-size guardrails.
- [x] 5.4 Run unit tests, compile checks, predeploy checks, and OpenSpec validation.
