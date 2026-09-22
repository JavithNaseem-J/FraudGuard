## Context

The repository currently has two serving paths. The legacy manual path loads baseline artifacts produced by a single-CSV pipeline, while the newer batch transaction path loads a promoted transaction model package. Even in transaction mode, application startup and predeploy checks still load or require the baseline artifacts. The legacy CSV is also still named in the dataset registry, DVC stages, configuration, tests, and historical notebook code.

The transaction candidate is locally verified, but its artifact directory is intentionally git-ignored. A clean Render checkout therefore cannot reproduce the local deployment. The existing GitHub Actions workflow targets AWS ECR and a self-hosted runner, not Render. Supabase persistence and Upstash rate limiting have application adapters, but account provisioning, migrations, release artifact delivery, and deployed smoke verification remain external steps. Observability currently defines the desired Evidently behavior but lacks a complete scheduled report workflow.

Constraints include free-tier service limits, private transaction data, no raw data or secrets in git, deterministic model/version selection, fail-fast readiness, and the need to preserve historical benchmark evidence without keeping the legacy dataset in the production path.

## Goals / Non-Goals

**Goals:**

- Make a clean repository checkout deploy the transaction model to Render without legacy data or baseline artifacts.
- Deliver immutable, versioned model bundles from private Supabase Storage with checksum verification and atomic activation.
- Make CI reproduce imports, tests, OpenSpec validation, artifact validation, and container startup before deployment.
- Provision and document Supabase, Upstash, Render, and Evidently workflows without exposing secrets.
- Produce operational evidence for drift, delayed-label performance, deployment health, and rollback readiness.
- Remove the legacy CSV from active configuration, dataset registry, DVC, serving, and deployment dependencies while retaining historical metrics as documentation.

**Non-Goals:**

- Retraining or materially changing the promoted transaction model in this change.
- Committing raw transaction data or binary model artifacts to git.
- Building end-user identity, multi-tenant authorization, or a paid observability platform.
- Guaranteeing high availability or an SLA on free-tier infrastructure.
- Deleting historical benchmark reports needed to explain prior model comparisons.

## Decisions

### 1. Supabase Storage is the artifact release source

Each release will be an immutable private object prefix containing the model, threshold, metadata, feature audit, and a manifest with filenames, sizes, SHA-256 checksums, schema version, model version, and creation time. Runtime configuration will select a release identifier, not a mutable `latest` object. Startup will download into a temporary release directory, verify every declared file, validate the candidate package, and atomically activate the directory only after all checks pass.

Alternatives considered:

- Commit model binaries to git: simple but increases repository size, weakens release separation, and conflicts with existing artifact exclusion.
- Render persistent disk: workable for one service, but manual population and instance coupling make reproducibility and rollback weaker.
- DVC remote pull: technically valid but adds credentials and tooling to the runtime image; Supabase Storage reuses the selected platform and is simpler for this release.

### 2. Transaction serving becomes the production runtime

Production startup will initialize only the configured transaction model. The manual baseline path may remain temporarily available for local historical demonstration, but it will be disabled by default and cannot participate in production readiness. The legacy dataset registry entry, DVC stages, configuration paths, and assertions will be removed or moved to explicitly historical documentation.

Alternatives considered:

- Keep dual startup: preserves compatibility but retains the exact missing-artifact coupling that blocks clean deployment.
- Delete all historical baseline evidence: removes confusion but loses useful comparison context.
- Selected approach: retire executable legacy dependencies while keeping clearly labeled historical results.

### 3. GitHub Actions gates a Render deployment

CI will install the repository package so imports do not depend on a manually supplied `PYTHONPATH`. It will run formatting/lint checks, tests, OpenSpec validation, compilation/import checks, predeploy validation with a fixture artifact bundle, container build, and container health smoke tests. Deployment will use a protected Render deploy hook or Render's Git integration only after CI succeeds. A post-deployment job will poll liveness/readiness and exercise an authenticated transaction prediction using non-sensitive fixture input.

The existing AWS ECR/self-hosted-runner workflow will be removed because it does not match the selected architecture. Render native auto-deploy without CI gating was considered but rejected because it can publish revisions before repository and artifact checks complete.

### 4. Evidently runs as a bounded batch monitoring workflow

Monitoring will extract sanitized feature summaries, prediction scores, decisions, model versions, and joinable delayed labels from Supabase. It will compare a versioned approved reference profile with a bounded production window, generate machine-readable metrics plus an HTML report, and store report metadata with the evaluated model/reference/window identifiers. Raw transaction payloads and provider secrets will not appear in reports.

An always-on monitoring service was considered but rejected for free-tier cost and operational complexity. Langfuse was not selected because the deployed system is tabular ML rather than an LLM application.

### 5. Failure behavior is explicit

- Artifact download, integrity, schema, or compatibility failure causes readiness to fail and prevents prediction traffic.
- A previously verified local artifact may be used only when it matches the configured release identifier and manifest.
- Supabase prediction persistence retains its documented serving fallback, but deployment smoke tests must expose the degraded state.
- Upstash follows the configured outage policy and emits sanitized telemetry.
- Monitoring failure does not stop serving; it produces a failed monitoring-run record and operator-visible exit status.

### 6. Promotion and rollback are release-ID operations

Promotion uploads an immutable bundle, verifies it independently, records the release in Supabase, updates Render's selected release identifier, deploys, and runs smoke tests. Rollback selects the prior verified release identifier and redeploys; it never mutates an existing release object.

## Data Flow

1. A verified transaction model package is assembled locally or in CI from approved artifacts.
2. The publisher creates and validates the manifest, uploads the private immutable release, and records release metadata.
3. Render starts with a configured release identifier and server-only Supabase credentials.
4. The bootstrapper downloads and verifies the release into temporary storage, atomically activates it, and initializes the transaction pipeline.
5. Readiness succeeds only after artifact validation; authenticated requests then pass request limits and Upstash rate limiting before inference.
6. Prediction and feedback metadata are written to Supabase when configured.
7. A scheduled/manual Evidently job reads bounded sanitized monitoring inputs and writes versioned report outputs and run metadata.

## Testing Strategy

- Unit tests cover manifest validation, checksum mismatch, path traversal rejection, atomic activation, cached-release matching, and rollback selection.
- Runtime tests prove transaction mode starts without baseline artifacts and fails readiness for missing or invalid selected releases.
- Dataset tests prove the active registry and DVC pipeline contain no legacy CSV dependency.
- CI tests run from a clean checkout without an externally supplied `PYTHONPATH`.
- Container tests exercise liveness, readiness, authentication, request limits, and fixture prediction.
- Monitoring tests use synthetic sanitized fixtures for drift, delayed-label joins, empty windows, and report metadata.
- Post-deployment smoke tests verify Render readiness, authenticated prediction, Supabase persistence, and rate-limit behavior without logging secrets or raw payloads.

## Risks / Trade-offs

- [Free-tier services may sleep, throttle, or change quotas] -> Document limits, use bounded retries, and distinguish cold-start delay from readiness failure.
- [Using Supabase service-role credentials for storage increases secret impact] -> Keep credentials server-only, use a private bucket, restrict object prefixes where practical, and never expose values in diagnostics.
- [Remote artifact startup adds latency and another failure mode] -> Cache by immutable release ID, validate cached manifests, and fail readiness instead of serving an unknown model.
- [Removing dual startup may break legacy demonstrations] -> Keep historical evidence and provide an explicit local-only migration note; production endpoints use the transaction contract.
- [Drift reports can expose sensitive distributions] -> Store only approved summaries, restrict report access, and exclude raw payload values.
- [A deployment hook is a privileged secret] -> Store it only in GitHub Actions secrets and restrict deployment to the protected branch/environment.

## Migration Plan

1. Repair package installation and CI so a clean checkout passes tests and OpenSpec validation.
2. Add artifact manifest, publisher, downloader, integrity validation, and local fixture tests.
3. Create the private Supabase Storage bucket and apply database migrations; upload a verified transaction release.
4. Decouple transaction startup/readiness/predeploy from baseline artifacts and remove active legacy CSV references.
5. Update Render configuration to transaction mode and the immutable release identifier; configure server-only secrets.
6. Configure Upstash and verify the selected outage policy.
7. Add and run Evidently monitoring against synthetic fixtures, then a bounded production window after deployment.
8. Deploy through the gated workflow and run post-deployment smoke tests.
9. If validation fails, set Render back to the prior verified release identifier and redeploy.

## Open Questions

No architecture question blocks implementation. Provider project identifiers, URLs, tokens, and deploy hooks must be supplied as external secrets during the apply/deploy phase and are intentionally absent from the repository.
