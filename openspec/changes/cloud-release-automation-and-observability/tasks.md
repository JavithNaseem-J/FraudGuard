## 1. Clean-checkout and legacy dependency tests

- [x] 1.1 Add regression tests proving production transaction startup, readiness, and predeploy validation do not require baseline artifacts or the retired legacy CSV.
- [x] 1.2 Add a repository check that fails when active configuration, dataset registry, DVC stages, serving code, CI, or deployment files reintroduce the retired legacy CSV path.
- [x] 1.3 Configure package installation and pytest so tests pass from a clean checkout without a manually supplied `PYTHONPATH`.

## 2. Retire the active legacy pipeline

- [x] 2.1 Remove the retired legacy dataset from the executable dataset registry and replace legacy comparison behavior with clearly labeled historical evidence.
- [x] 2.2 Remove legacy CSV inputs and outputs from active configuration, DVC stages, lock state, tests, and production documentation while preserving relevant historical metrics.
- [x] 2.3 Refactor application startup so production transaction mode initializes only the transaction pipeline and does not construct the baseline predictor.
- [x] 2.4 Update readiness, health metadata, prediction routing, and predeploy checks for independent transaction-mode operation and an explicit local-only legacy mode if retained.

## 3. Versioned model artifact delivery

- [x] 3.1 Define and test the immutable release manifest schema with release ID, artifact schema, model version, required files, byte sizes, SHA-256 checksums, and creation metadata.
- [x] 3.2 Implement safe local bundle assembly and validation, including required-file checks, checksum calculation, path traversal rejection, and secret-safe diagnostics.
- [x] 3.3 Implement private Supabase Storage publication under immutable release prefixes and persist corresponding model-release metadata without uploading raw data.
- [x] 3.4 Implement startup download to a temporary directory, integrity and candidate-package validation, atomic activation, and verified cache reuse keyed by release ID.
- [ ] 3.5 Add rollback selection support for previously verified release IDs and tests proving failed activation leaves the prior verified release untouched.
- [x] 3.6 Extend settings and predeploy validation for storage bucket, release ID, cache directory, timeout, and retry configuration without printing credential values.

## 4. Supabase, Upstash, and Render configuration

- [x] 4.1 Add idempotent Supabase migration or setup documentation for private model storage, release metadata, monitoring runs, and required access policies.
- [x] 4.2 Update Render configuration to transaction production mode, immutable release selection, server-only provider secrets, appropriate health checks, and no bundled untracked artifacts.
- [x] 4.3 Document Upstash creation, environment configuration, distributed enforcement verification, and the chosen outage policy.
- [x] 4.4 Update Docker behavior so a clean build succeeds without a local `artifacts/` directory and the non-root runtime can write only to its configured artifact cache.

## 5. CI/CD release automation

- [x] 5.1 Replace the AWS ECR and self-hosted deployment jobs with clean-checkout CI and protected Render deployment stages.
- [x] 5.2 Add deterministic dependency installation, formatting/lint checks, unit and integration tests, compilation/import checks, strict OpenSpec validation, and fixture-based predeploy validation.
- [ ] 5.3 Add container build and local smoke tests for liveness, readiness, API authentication, request limits, and a non-sensitive transaction fixture.
- [x] 5.4 Add a gated Render deploy trigger and bounded post-deployment smoke tests using protected credentials, with a failed-release status and documented rollback action.
- [ ] 5.5 Confirm workflow logs and artifacts contain no provider secret, API key, signed URL, or raw transaction payload.

## 6. Evidently operational monitoring

- [x] 6.1 Define an approved sanitized monitoring schema and versioned reference profile compatible with the production transaction feature contract.
- [ ] 6.2 Implement a bounded Supabase extraction workflow for feature summaries, scores, decisions, release metadata, and joinable confirmed feedback.
- [x] 6.3 Implement Evidently drift report generation with release, reference, window, row-count, and insufficient-data metadata in machine-readable and HTML outputs.
- [x] 6.4 Implement delayed-label performance reporting with label coverage, threshold metrics, PR-AUC when valid, calibration evidence, and cost-weighted loss without treating unlabeled rows as negatives.
- [ ] 6.5 Persist sanitized monitoring-run status and report metadata, and add tests for success, no-data, insufficient-data, failed extraction, and delayed-label scenarios.

## 7. Documentation and operator runbooks

- [x] 7.1 Update README and cloud documentation with clean-checkout setup, Supabase Storage publication, migrations, Render deployment, Upstash verification, and Evidently execution.
- [x] 7.2 Add promotion and rollback runbooks using immutable release IDs, including readiness failure recovery and prior-release restoration.
- [x] 7.3 Document free-tier cold starts, quotas, scheduling limitations, retention, security boundaries, and the difference between local verification and deployed evidence.
- [x] 7.4 Use professional transaction-data wording throughout active docs and remove claims that the retired legacy dataset is a production dependency.

## 8. End-to-end verification

- [ ] 8.1 Run the complete test suite, strict OpenSpec validation, compile/import checks, and production transaction predeploy checks from a clean environment.
- [ ] 8.2 Publish and independently verify a transaction model release without recording secrets or raw data in repository output.
- [ ] 8.3 Verify the production container starts without baseline artifacts and serves authenticated transaction predictions from the selected release.
- [ ] 8.4 Execute deployed Render smoke checks for readiness, prediction, Supabase persistence, Upstash rate limiting, and rollback to a prior verified release where credentials are available.
- [ ] 8.5 Generate representative Evidently drift and delayed-label reports from sanitized fixtures and explicitly record any external provider checks that could not be executed locally.
