## Why

FraudGuard now has a cloud-ready API and a feature-flagged transaction candidate path, but it should not be exposed publicly without basic access control, request guardrails, deploy preflight checks, and clear artifact delivery guidance. This change turns the current demo into a safer production-shaped deployment target for free-tier platforms.

## What Changes

- Add API-key protection for prediction, batch prediction, and feedback endpoints.
- Add request-size and batch-size guardrails with safe defaults.
- Extend deployment preflight checks for Render/Supabase/Upstash/candidate artifact configuration.
- Document a safe artifact delivery strategy for local and Render deployments.
- Update README/cloud docs with verified metrics, setup steps, limitations, and safe claims.
- Preserve local development defaults so tests and demos can run without cloud secrets.

## Capabilities

### New Capabilities

- `api-security-and-deploy-hardening`: Covers API-key access control, request guardrails, deployment preflight, artifact delivery guidance, and production-readiness documentation.

### Modified Capabilities

- `cloud-service-runtime`: Add secure runtime configuration and deployment preflight expectations.
- `distributed-rate-limiting`: Add request guardrails that complement Upstash/local rate limiting.
- `cloud-persistence-and-audit`: Add sanitized audit events for denied requests and deployment checks.
- `fraud-observability`: Add security/deployment telemetry expectations without logging raw payloads or secrets.

## Impact

- Affected API: prediction, batch transaction prediction, feedback, health/readiness behavior.
- Affected config: new environment variables for API auth and request limits.
- Affected docs: README, cloud-native deployment guide, Render setup notes.
- Affected scripts: provider predeploy checks.
- No raw dataset changes.
- No paid-provider requirement.
- No full user login system in this change.
