## 1. OpenSpec and Architecture

- [x] 1.1 Validate the OpenSpec proposal, design, tasks, and spec deltas before implementation.
- [x] 1.2 Confirm the first benchmark dataset training, test, and target file paths, schema, and license notes without committing raw data.
- [x] 1.3 Choose initial false-positive and false-negative cost defaults and document them as configurable assumptions.

## 2. Cloud Runtime

- [x] 2.1 Add environment-driven application settings for app environment, artifact root, provider URLs, and feature flags.
- [x] 2.2 Load prediction artifacts once at startup or lifespan initialization and expose readiness based on loaded artifact state.
- [x] 2.3 Split liveness and readiness health responses so Render can detect app process health separately from model readiness.
- [x] 2.4 Return a safe JSON prediction response with a server-generated prediction ID and no transaction payload in result-page URLs.
- [x] 2.5 Add Render deploy configuration and update the Docker/runtime command to use provider-injected ports.

## 3. Supabase Persistence and Audit

- [x] 3.1 Add Supabase SQL migrations for prediction requests, prediction feedback, model releases, and audit events.
- [x] 3.2 Add indexes and constraints for model version, created timestamp, request ID, and feedback joins.
- [x] 3.3 Add a server-side persistence adapter that uses Supabase when configured and degrades explicitly when absent.
- [x] 3.4 Persist prediction request metadata, model release, threshold, score, decision, and calibrated-score flag after successful inference.
- [x] 3.5 Add feedback capture behavior that links delayed labels or user corrections to prediction IDs.

## 4. Upstash Rate Limiting

- [x] 4.1 Add Upstash Redis configuration for distributed API rate limiting.
- [x] 4.2 Preserve local in-memory limiting as the development fallback when Upstash is not configured.
- [x] 4.3 Define outage behavior for Redis failures and cover it with tests or documented smoke checks.

## 5. Dataset Benchmarking and Cost-Aware ML

- [x] 5.1 Add a split-aware dataset registry/config convention for the current baseline dataset and one external benchmark dataset with training, test, and target artifacts.
- [x] 5.2 Add dataset validation for required target joins, provider train/test split integrity, feature availability, duplicates, class balance, and leakage-risk columns.
- [x] 5.3 Update threshold selection to support cost-weighted loss using training-only validation or out-of-fold scores.
- [x] 5.4 Update evaluation output to include cost-weighted metrics alongside AP, ROC-AUC, Brier score, precision, recall, F1, support, prevalence, and confusion matrix.
- [x] 5.5 Persist cost assumptions and selected operating policy in model metadata.

## 6. Observability and Monitoring

- [x] 6.1 Add structured service logs that include request ID, prediction ID, model version, latency, and outcome without sensitive payloads.
- [x] 6.2 Add Evidently report generation for unlabeled drift using prediction records and a reference dataset.
- [x] 6.3 Add delayed-label performance report support when feedback labels become available.
- [x] 6.4 Document Grafana Cloud setup for logs/metrics and Evidently OSS report storage.

## 7. Documentation and Verification

- [x] 7.1 Document Supabase, Upstash, Render, Grafana, and Evidently setup steps with required environment variables and no secret values.
- [x] 7.2 Document free-tier limitations and the enterprise upgrade path for uptime, backups, HA Redis, protected environments, and compliance.
- [x] 7.3 Add or update tests for runtime settings, prediction response contract, persistence fallback, rate limiting fallback, dataset registry, and cost-weighted threshold selection.
- [x] 7.4 Run feasible verification: OpenSpec validation, unit tests, import/syntax checks, and API smoke checks.
