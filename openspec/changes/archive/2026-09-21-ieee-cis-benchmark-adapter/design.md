## Context

The workspace contains `data/train_transaction.csv`, `data/train_identity.csv`, and `data/test_identity.csv`. There is no public test target file. For IEEE-CIS-style fraud data this is normal: public test rows are often meant for competition submission or unlabeled scoring, not offline performance measurement.

The current FraudGuard pipeline expects a relatively small single CSV with the target column in place. The benchmark adapter should avoid forcing the IEEE-CIS data into that old shape too early. It should create a clean, explicit preparation path that keeps labeled evaluation honest.

## Goals / Non-Goals

**Goals:**

- Prepare an IEEE-CIS labeled modeling table from local transaction data and optional identity data.
- Preserve target semantics: `isFraud` is the positive class.
- Create deterministic internal splits from labeled training rows for benchmark model selection, threshold selection, and final evaluation.
- Report cost-weighted fraud metrics and class imbalance metrics.
- Keep raw data out of git and keep public test rows unlabeled.

**Non-Goals:**

- No cloud deployment changes.
- No model replacement in the serving API until benchmark performance is verified.
- No metrics from unlabeled public test rows.
- No raw IEEE-CIS data committed to the repository.

## Decisions

### Decision: Metrics come from labeled training rows only

Use `train_transaction.csv` as the labeled benchmark source. Split it deterministically into training, validation/threshold, and final benchmark test partitions. Public test files can be transformed and scored later, but they cannot produce precision, recall, AP, or cost-weighted loss without labels.

Alternative considered: treat public test rows as the final test set. That would produce fake metrics because labels are absent.

### Decision: Identity join is optional and explicit

Join `train_identity.csv` to `train_transaction.csv` by `TransactionID` only when configured. Missing identity rows must be handled as missing feature values rather than dropping transaction rows silently.

Alternative considered: require identity data for every row. That would throw away labeled transactions and bias the benchmark.

### Decision: Start with tabular pipeline compatibility

The adapter should emit prepared CSVs or dataframes compatible with the existing sklearn pipeline where feasible. Dataset-specific feature selection and column typing should be explicit so serving features are not accidentally mixed with benchmark-only fields.

Alternative considered: build a completely separate modeling stack. That may be useful later, but first we need comparable, defensible metrics.

## Risks / Trade-offs

- IEEE-CIS is large -> implementation should support sampled smoke validation and full-run mode separately.
- Identity columns are sparse -> preprocessing must tolerate missing values and high-cardinality categories.
- Benchmark features differ from the current API features -> benchmark success does not automatically mean the deployed API can use the same model.
- Public test labels are unavailable -> documentation must clearly distinguish offline benchmark metrics from unlabeled scoring.

## Migration Plan

1. Add adapter config for IEEE-CIS local paths and split parameters.
2. Implement schema and leakage validation on sampled rows first, then full data.
3. Implement deterministic labeled splits from `train_transaction.csv`.
4. Add optional identity joins.
5. Generate benchmark metrics and reports from the held-out labeled split.
6. Keep serving model promotion as a later OpenSpec change.

## Open Questions

- Should the first full benchmark run use all rows or a memory-bounded sample to fit local hardware?
- Which columns should be excluded as identifiers versus usable categorical features beyond `TransactionID`?
