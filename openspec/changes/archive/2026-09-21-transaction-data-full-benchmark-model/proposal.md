## Why

The current transaction benchmark smoke model proves the evaluation path works, but its recall and F1 are too weak for production fraud use. We need a stronger, leakage-safe tabular benchmark before deciding whether any model should replace or influence the deployed API.

## What Changes

- Add a stronger transaction benchmark modeling pipeline using LightGBM when available, with the numeric logistic smoke baseline preserved for comparison.
- Add feature filtering, categorical handling, missing-value handling, and memory-bounded sample/full modes.
- Select thresholds using cost-weighted validation scores, not accuracy or the public unlabeled test data.
- Report model comparison metrics, including AP, ROC-AUC, precision, recall, F1, confusion matrix, prevalence, Brier score, and cost-weighted loss.
- Keep serving promotion explicitly out of scope until benchmark evidence is strong enough.
- Keep public docs and reports using professional “transaction data” language.

## Capabilities

### New Capabilities

- `transaction-benchmark-modeling`: Stronger transaction-data benchmark training, comparison, evaluation, and promotion decision reporting.

### Modified Capabilities

- None.

## Impact

- Affected ML areas: transaction benchmark adapter, benchmark training/reporting utilities, tests, and benchmark documentation.
- Affected artifacts: ignored benchmark outputs under `artifacts/benchmark/transaction_data/`.
- No cloud provider changes and no serving model replacement in this change.
