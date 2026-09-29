# README evidence audit

Audit date: 2026-09-29. Scope: the current working tree, including pre-existing edits to `src/FraudGuard/cloud/artifacts.py` and `tests/test_core.py`. No application, training, test, Docker, deployment, or external API command was run for this audit.

## Repository coverage

- **154 relevant text/JSON files discovered and inspected:** root configuration and README (15), GitHub workflows (2), frontend source/configuration/lock (44), scripts (7), Python package (16), Supabase migrations (4), tests (2), local documentation (1), OpenSpec documentation (38), and generated JSON evidence (23). The two dependency locks and frontend lock were parsed for their full package inventories; generated reports were parsed in full and their claim-bearing fields inspected. Source files that exceeded one response were read in chunks.
- **Large data inspected by schema/metadata only:** `data/train_transaction.csv`, `data/test_transaction.csv`, `frontend/public/sample_transactions.csv`, eight generated split CSVs, and the duplicate frontend build sample. The source files are about 683 MB and 613 MB. Their headers were inspected, and the saved preparation report supplied row counts, class balance, split ranges, and provenance. The sample CSV has exactly the same 392 feature names as the approved release, in a different order; inference reorders them.
- **Skipped categories (15):** `.git/` history objects (used `git log/show` instead); `frontend/node_modules/` (managed dependencies); generated `frontend/dist/` and `artifacts/frontend-build-smoke/` (compiled assets); `graphify-out/` (generated graph/cache); `.dvc/cache/` and `.dvc/tmp/` (cache/locks); `.mypy_cache/`, `.ruff_cache/`, `.pytest_temp/`, `__pycache__/`, and `src/fraudguard.egg-info/` (generated caches/metadata); `logs/` (runtime output); `.vscode/` (local editor state); private `.env` (secrets, intentionally unread); model `.joblib` files and full large CSV contents (binary/large data; manifests, schemas and report metadata inspected). The generated `pytest-run-temp/` directory could not be enumerated due to access denial; it is test temporary output and contains no required source.
- **Truncated/unreadable relevant files:** none. An initial combined tool response was truncated, so affected source groups were read again separately. No relevant source was skipped for size.
- **Git history:** the latest 50 commits were read; relevant changes were cross-checked with current files, including `59dbde8` (single service simplification), `fc6ab83` (train-fitted feature engineering), `640eefa` (changed promotion gates), `826d2c1` (full-data metrics), `2722640` (public dashboard delete), and `4226cab` (cleanup).

## Project summary and runtime flow

FraudGuard is a portfolio transaction fraud classifier with a chronological offline LightGBM evaluation workflow and a same-origin React/FastAPI scoring demo.

**Training:** labeled `train_transaction.csv` -> contract check and exact-row deduplication -> stable `TransactionDT` group split -> numeric logistic baseline and eight LightGBM pipelines fitted on training -> validation-only candidate/threshold choice -> frozen later holdout -> promotion gates -> evaluated bundle and checksum manifest. The unlabeled public test file is only used for schema/inference checks.

**Serving:** browser CSV/JSON -> React schema check -> FastAPI request/batch/rate guards -> release manifest and `TransactionPipeline` -> ordered features and fitted preprocessing/model -> scores plus stored threshold -> sanitized prediction records in Supabase, or bounded process memory without credentials -> dashboard aggregate response. `GET /ready` depends on model loading; `GET /live` does not. The public `DELETE /dashboard` and `POST /dashboard/clear` currently erase every stored prediction.

## Technology stack

Python 3.11-3.13 by `pyproject.toml`; FastAPI, Pydantic 2, pandas, scikit-learn, LightGBM, joblib, NumPy; React 18.3.1, TypeScript 5.9.3, Vite 8.3.0, Papa Parse 5.7.0; optional Supabase PostgREST/Storage and Upstash Redis REST; Docker (Python 3.11 and Node 20 stages), Render blueprint, GitHub Actions; DVC and Evidently in development tooling. Versions are from `pyproject.toml`, Python locks, `frontend/package-lock.json`, and `Dockerfile`. No model provider or LLM is involved.

## Active versus inactive code and real versus mock versus toy

- **Active:** `app.py`, `src/FraudGuard/{cloud,data,pipeline,monitoring,utils}`, `frontend/src`, scripts, migrations, CI/Docker/Render configurations. The DVC stage is a 75,000-row diagnostic, not the full release workflow. Evidently is an offline command, not automatic production monitoring.
- **Historical or stale:** `artifacts/benchmark/ieee_cis/` identity-joined sample, `artifacts/transform/split/` old 10-column data, the older `artifacts/benchmark/transaction_data/model/` sample candidate, and root `openspec/specs/` requirements that still mention authentication, feedback, identity joins, and other retired paths. These are excluded from current-capability claims. OpenSpec change documents also describe intentions that differ from current code.
- **Real:** local labeled/unlabeled transaction CSVs, the saved full-data report and approved release bundle, actual API/React routes, SQL migrations, CI and deployment workflow definitions. The README's pre-existing live URL is a declared demo link; its availability and exact deployed model were not verified here.
- **Synthetic/toy:** `scripts/build_ci_model_fixture.py` builds a tiny synthetic logistic model for container smoke tests; `tests/test_core.py` uses synthetic rows and mocked provider calls. The frontend sample CSV contains real-shaped unlabeled example rows, but it is demo input and is not model-quality evidence.
- **Optional/external:** Supabase persistence and private artifact storage need server credentials; otherwise predictions are stored only in bounded process memory. Upstash limits are used only when configured; otherwise counting is per process.

## Verified metrics and numeric evidence

The following values are **verified against saved local JSON evidence**, not independently reproduced by running training in this session. Generated data and artifacts are ignored by Git, so a clean GitHub clone does not expose this evidence by default.

| Metric | Value | Evidence |
|---|---:|---|
| Labeled source rows | 590,540 | `artifacts/benchmark/transaction_data/preparation_report.json` `source_rows` |
| Train / validation / holdout rows | 413,378 / 88,581 / 88,581 | Same report, `partitions` |
| Input features, numeric / categorical | 392, 378 / 14 | `artifacts/benchmark/transaction_data/evaluated-model/feature_audit.json` |
| Selected LightGBM candidate / threshold | `lgbm-05` / 0.0282619636 | `artifacts/benchmark/transaction_data/strong_benchmark_report.json` `strong_benchmark.tuning`, `evaluated-model/threshold.json` |
| Holdout AP / recall / precision | 0.539642 / 0.780733 / 0.199718 | `artifacts/benchmark/transaction_data/strong_benchmark_report.json` `strong_benchmark.metrics` |
| Holdout mean modeled cost, LightGBM / numeric logistic | 0.261512 / 0.434190 | Same report, `strong_benchmark.metrics.cost_weighted` and `smoke_benchmark.metrics.cost_weighted` |
| Holdout TP / FP / FN / TN | 2,407 / 9,645 / 676 / 75,853 | Same report, `strong_benchmark.metrics.confusion_matrix` |
| False-positive / false-negative cost weights | 1 / 20 | `src/FraudGuard/utils/costs.py`, `evaluated-model/threshold.json` |
| Promotion gates passed | 8 of 8 in saved report | `strong_benchmark_report.json` `strong_benchmark.promotion_gates` |
| Manifest integrity | four payload hashes and sizes match locally | `artifacts/releases/tx-20260927-001/active/manifest.json` and four local payloads, recomputed SHA-256 in this audit |

No bank loss, savings, precision at a fixed review capacity, latency percentile, test coverage, or production drift metric is established. The modeled cost comparison uses the declared 1:20 assumption. The output score is explicitly marked uncalibrated in release metadata.

## Engineering capabilities and evidence

| Area | Evidence and boundary |
|---|---|
| Tests | `tests/test_core.py` covers split boundaries, artifact contracts, request guards, local dashboard, and monitoring sanitization. Frontend `npm test` is a source/sample contract script, not browser interaction tests. Tests were inspected, not run here. |
| CI and deployment | `.github/workflows/ci.yml`, `render-deploy.yml`, `Dockerfile`, `render.yaml`: lint/type/tests/frontend/container smoke and exact-commit Render workflow exist. Current live release/commit was not verified. |
| Validation and release | `transaction_benchmark.py`, `transaction_pipeline.py`, `cloud/artifacts.py`: ordered feature contract, release gates, manifest hashes, startup readiness, manifest-last upload. |
| Persistence | `cloud/persistence.py`, migrations `001`-`004`: sanitized records via server-side Supabase; bounded local-memory fallback. Provider write failure can still leave a successful score with `persisted_count=0`. |
| Logging/monitoring | `app.py` logs request IDs and model/persistence metadata. `monitoring/evidently_reports.py` supports manual sanitized output drift comparisons. No scheduled production drift or delayed-label evaluation. |
| Rate limiting/security | `cloud/rate_limit.py`, `app.py`: anonymous scoring, byte/batch guards, optional Upstash with configurable fail-open. `client_key` trusts `x-forwarded-for`; no user authentication or tenant isolation. Public dashboard deletion is a material integrity risk. |

## Engineering decisions

1. **Chronological time-group split** instead of a random split for later-period evidence. The code preserves equal timestamp groups; trade-off is lower and potentially more realistic metrics. Evidence: `src/FraudGuard/data/transaction_benchmark.py`, full-data report, design in `openspec/changes/full-data-transaction-model-release/design.md`.
2. **Train-fitted feature engineering inside the LightGBM pipeline** so frequency/amount maps are learned from train rows; trade-off: the unseen-group amount-ratio fallback uses the current transform batch mean, making that feature potentially batch-dependent. Evidence: `TransactionFeatureEngineer`; commit `fc6ab83`.
3. **Validation-selected 1:20 threshold and explicit release gates** to separate ranking, decision cost, and publication. The 1:20 ratio is an assumption. Commit `640eefa` explicitly relaxed the AP gate from 0.55 to 0.50 and cost gate from 0.28 to 0.30 after describing holdout AP near 0.54 and cost near 0.26. Thus the saved holdout was one-time within its run but not prospectively untouched across project development. Evidence: benchmark source/report and that commit.
4. **Immutable model bundle with manifest-last upload** and checksum-validated download. Trade-off: release storage/credentials and compatibility shims must be managed; no claim of automatic rollback. Evidence: `cloud/artifacts.py`, `scripts/publish_model_release.py`, `pipeline/transaction_pipeline.py`.
5. **One same-origin API/UI container with optional cloud services** for a bounded public demo. Trade-off: anonymous public endpoints and local per-process fallback have limited isolation. Evidence: `Dockerfile`, `app.py`, `frontend/src/services/api.ts`, `render.yaml`; commit `59dbde8`.

## Mermaid mapping

| README node | Implementation evidence |
|---|---|
| Labeled CSV | `data/train_transaction.csv`, benchmark config |
| Contract and time split | `validate_transaction_data_contract`, `split_labeled_transaction_data` |
| Train-fitted models | `_baseline_pipeline`, `_lightgbm_pipeline`, `_tune_lightgbm_candidate` |
| Validation selection | `_tune_lightgbm_candidate`, `select_cost_weighted_threshold` |
| Later holdout and gates | `run_transaction_strong_benchmark`, `_promotion_gates` |
| Model bundle | `_write_model_artifacts`, `cloud/artifacts.py` |
| Browser console | `frontend/src/pages/OverviewPage.tsx`, `ScoreTransactionsPage.tsx` |
| FastAPI | `app.py` |
| Request guards | `enforce_prediction_request_size`, `predict_transaction_batch`, `CloudRateLimiter` |
| Loaded scoring pipeline | `_load_transaction_model`, `TransactionPipeline.predict_rows` |
| Prediction store | `SupabasePersistence.persist_predictions` |
| Dashboard snapshot | `app.dashboard`, `_dashboard_snapshot`, `frontend/src/services/dashboardStore.ts` |

## Strongest hiring signals

Chronological holdout and explicit baseline; validation-selected cost threshold with sensitivity analysis; feature/order and artifact integrity contracts; full API-to-UI flow; bounded anonymous demo and CI/container/deployment workflow definitions.

## Weaknesses and inconsistencies

1. **Public deletion:** `DELETE /dashboard` and `POST /dashboard/clear` have no authentication and delete all records, while current OpenSpec simplification docs describe owner-only deletion. The UI's `Clear` has an `onClick`, so the `onDoubleClick` handler does not enforce a confirmation gesture. Do not claim protected audit history or confirmed deletion safeguards.
2. **Inference and evidence gaps:** unseen `card1`/`addr1` amount-ratio fallback uses the mean of whichever batch is transformed, so a row may score differently alone versus in a batch. Scores are uncalibrated, no delayed labels or live monitoring demonstrate post-deployment quality, and the generated full-data report is unavailable in a clean Git clone.
3. **Reproducibility/deployment drift:** `dvc.lock` records older hashes/sizes for benchmark source and script than the current working tree; `dvc.yaml` runs a diagnostic cap, not the full release. Root OpenSpec specs and change specs state 0.70 AP/0.20 cost gates, while active source/report use 0.50/0.30. Commit `640eefa` used observed holdout performance to relax the gates, so gate passage is not independent prospective validation. The `.env.example` local root names an older sample candidate, and the live demo's release/commit was not verified.

## Interview questions

1. **How do you prevent train/validation/holdout leakage, including learned frequency features?** Answered well by README.
2. **Why is the operating threshold 0.028 rather than 0.5, and how defensible is the 1:20 error-cost ratio?** Answered well by README.
3. **Can the same unseen-category transaction receive different scores when scored alone versus in a batch?** Answered well by README as a limitation; current implementation has no fix.
4. **What stops an anonymous visitor from deleting everyone else's dashboard predictions?** Answered well by README as a limitation; current implementation has no access control.
5. **Can an interviewer reproduce the headline result and verify the currently deployed release from a clean clone?** Partially answered: commands and evidence paths are documented, but data/artifacts are ignored and live acceptance was not independently verified.

## Missing information / author follow-ups

1. TODO for author: provide a shareable, redacted full-data report (or immutable public evidence link) so GitHub reviewers can inspect the reported metrics without the ignored local artifacts.
2. TODO for author: verify the hosted URL, exact deployed commit, model release ID, and 392-feature schema after deployment; current OpenSpec task records still mark live acceptance steps incomplete.
3. TODO for author: provide intended deployment/ownership policy for the anonymous dashboard deletion endpoints, or separately authorize a code fix.
4. TODO for author: document dataset acquisition/provenance and any redistribution limits for reproducibility; the source CSVs are local and ignored.

No Author Context values were supplied. The current local `.env` was intentionally not read. No unresolved marker belongs in the public README.
