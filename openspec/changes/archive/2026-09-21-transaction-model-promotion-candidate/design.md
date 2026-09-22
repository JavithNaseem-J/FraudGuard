## Context

The transaction-data strong benchmark outperformed the numeric logistic smoke baseline on a bounded local sample. That score is a useful signal, but serving promotion would still be premature because the benchmark feature schema is wider than the current API input schema and the public test files do not include labels.

## Goals / Non-Goals

**Goals:**

- Export a reproducible, non-serving candidate package from the strong benchmark.
- Persist threshold, metrics, cost assumptions, feature list, and schema audit metadata.
- Run a larger bounded transaction-data benchmark to improve confidence in the candidate.
- Make the promotion decision explicit and conservative.

**Non-Goals:**

- Do not replace `artifacts/trainer` or the current API serving model.
- Do not compute metrics from public unlabeled test files.
- Do not introduce paid cloud services or remote deployment in this change.
- Do not destructively modify raw transaction data.

## Decisions

### Decision: Candidate package, not serving deployment

The benchmark runner will write a `candidate/` package under the benchmark output directory. This package can include the fitted pipeline, operating threshold, metadata, and feature audit, but it will mark `serving_promotion` as false.

### Decision: Promotion gates are evidence, not automatic approval

The benchmark will evaluate candidate metrics against explicit gates such as AP, recall, and cost-weighted loss. Passing gates means the candidate is worth a later serving-compatibility change; it does not update production artifacts.

### Decision: Audit schema risk now

The candidate metadata will record selected features, dropped columns, high-missing columns, categorical/numeric counts, and the fact that public unlabeled files were not used for metrics. This makes training-serving skew visible before deployment work starts.

## Risks / Trade-offs

- Larger local runs may take longer or consume memory; the runner will remain bounded by `sample_rows`.
- Feature names may be numerous; metadata will store enough detail for reproducibility while keeping the summary readable.
- Good internal metrics may still fail production-readiness if serving inputs cannot supply the same features.

## Validation Plan

1. Add unit coverage for candidate artifact creation and non-promotion behavior.
2. Run unit tests and compile checks.
3. Validate the OpenSpec change.
4. Run a larger bounded local transaction-data benchmark if local data is available.
