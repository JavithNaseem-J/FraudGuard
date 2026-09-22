## Context

FraudGuard is a FastAPI and scikit-learn fraud detection application with a DVC-backed training pipeline. The current repository has strong work in progress around leakage controls, inference artifact validation, and evidence-backed claims, but the serving layer remains local-demo oriented. It initializes prediction artifacts inside request paths, uses process-local rate limiting, exposes prediction input data through result-page query strings, and has no durable store for predictions, feedback, model release metadata, or audit events.

The target is not to pretend that free services are enterprise production. The target is a cloud-native foundation that can run on free tiers for demonstration, while making the production upgrade path explicit.

## Goals / Non-Goals

**Goals:**

- Make the service deployable on Render using environment-driven configuration and container-friendly behavior.
- Add deploy-ready configuration for Supabase and Upstash without committing secrets or creating provider resources automatically.
- Persist prediction, feedback, model release, and audit metadata in Supabase.
- Use Upstash Redis for distributed rate limiting when configured, with local fallback for development.
- Preserve the current dataset as the baseline and add a benchmark dataset slot for the already-downloaded external dataset.
- Move operating-policy decisions toward cost-weighted loss and report the business cost of false positives and false negatives.
- Add observability hooks and report generation that fit a tabular ML fraud app.

**Non-Goals:**

- No live cloud provisioning in this change.
- No paid-provider-only features as hard requirements.
- No claim that free Render, Supabase, or Upstash tiers provide enterprise uptime, backups, support, or compliance.
- No automated fraud blocking workflow; model outputs remain advisory until model quality and business policy are validated.
- No raw external dataset committed to git.

## Decisions

### Decision: Build a free-tier demo with an enterprise upgrade path

Use Render, Supabase, and Upstash as the first deploy target because they have approachable free tiers and clear upgrade paths. The application and docs must mark free-tier limits, including service sleeping, paused projects, storage limits, command quotas, and missing production guarantees.

Alternative considered: design immediately for paid cloud services such as Cloud Run plus managed Postgres and Redis. That is more production-realistic, but it blocks the user's free-platform goal.

### Decision: Supabase is the durable system of record

Store prediction requests, prediction feedback, model release metadata, and audit events in Supabase Postgres. Use server-side service credentials only in the FastAPI process. Add SQL migrations and indexes, but keep browser-facing auth out of scope for this first cloud foundation unless later requested.

Alternative considered: keep JSON/CSV logs or local SQLite. Those are simpler locally but do not support cloud-native history, auditability, or multi-instance behavior.

### Decision: Upstash owns distributed limits; the database owns durable idempotency

Use Upstash Redis for request-rate enforcement across Render instances when configured. Do not rely on Redis alone for permanent duplicate prevention. Durable uniqueness, prediction IDs, and audit history belong in Supabase.

Alternative considered: keep SlowAPI memory storage only. That is acceptable for local development, but a multi-instance deployment would have inconsistent limits.

### Decision: Model policy becomes cost-aware

Keep ranking metrics such as average precision, but choose and report the deployed threshold using explicit false-positive and false-negative costs. Persist those costs and the selected objective in model metadata so inference can explain which operating policy produced a decision.

Alternative considered: continue maximizing F1. F1 is useful but hides the asymmetric business cost of missed fraud versus manual review.

### Decision: Benchmark datasets are pluggable, external, and split-aware

Keep the current dataset as the reproducible baseline. Add a dataset registry/config pattern for an external benchmark dataset already downloaded by the user. The registry must support benchmark datasets delivered as labeled training data plus unlabeled public test files, or as separate `training`, `test`, and `target` files when labels are available. Because this workspace has no test target file, benchmark metrics must come from a split of the labeled training artifact. The repo records expected schema, path conventions, checksums when available, and benchmark metrics, but it does not commit raw data.

Alternative considered: replace the current dataset immediately. That would lose comparison evidence and make it harder to know whether the new dataset/model is genuinely better.

### Decision: Evidently for ML monitoring; Grafana for service observability

Use Evidently OSS for batch drift and performance reports on tabular prediction data. Use Grafana Cloud-compatible structured logs/metrics documentation for service telemetry. Langfuse is deferred unless the app later adds LLM workflows.

Alternative considered: Langfuse now. It is strong for LLM tracing, but this project is currently a tabular ML API.

## Risks / Trade-offs

- Free-tier hosting is not enterprise production -> docs must state limits and identify paid upgrades for uptime, backups, support, and compliance.
- Supabase service-role key exposure would be severe -> server-only environment variables, no frontend injection, and no secret values in logs or docs.
- Cost-weighted loss needs business assumptions -> use configurable defaults and require the chosen costs to appear in model metadata and evaluation output.
- Benchmark dataset schema may differ from the current dataset or arrive as separate training/test/target files -> add a split-aware dataset adapter/registry contract rather than forcing all datasets into one implicit schema.
- Drift/performance monitoring needs production labels, which are delayed in fraud -> support unlabeled drift now and delayed-label performance joins later.

## Migration Plan

1. Add settings and runtime behavior behind environment variables, preserving local defaults.
2. Add Supabase SQL migrations and a small persistence adapter that no-ops or degrades clearly when Supabase is not configured.
3. Add Upstash-backed limiter configuration with memory fallback for local development.
4. Add split-aware dataset registry and benchmark documentation without moving or committing raw data.
5. Add cost-weighted training/evaluation configuration and metadata fields.
6. Add Render deployment config and provider setup docs.
7. Verify locally with tests, OpenSpec validation, import checks, and API smoke tests where feasible.

Rollback is local and config-based: unset Supabase/Upstash environment variables to return to local fallback behavior, and deploy the previous Render version if a cloud deployment fails.

## Open Questions

- Resolved: the first benchmark dataset is an IEEE-CIS-style local dataset with labeled `data/train_transaction.csv`, identity side files, and no public test target.
- What default cost ratio should be used for false negatives versus false positives until real business costs are available?
- Should Supabase Auth be added in the next change, or should this first pass remain API-key/internal-service oriented?
