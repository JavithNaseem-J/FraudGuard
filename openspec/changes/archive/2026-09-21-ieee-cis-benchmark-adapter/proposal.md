## Why

FraudGuard now has a cloud-native foundation, but the large IEEE-CIS-style benchmark data needs a dedicated adapter before it can be used defensibly. The public test set has no target labels, so benchmark metrics must come from labeled training data rather than pretending unlabeled test rows can produce final evaluation scores.

## What Changes

- Add an IEEE-CIS benchmark adapter that reads local `train_transaction.csv`, `train_identity.csv`, and `test_identity.csv` files without committing raw data.
- Prepare labeled benchmark data from `train_transaction.csv`, optionally joining `train_identity.csv` by `TransactionID`.
- Create a deterministic internal train/validation/test split from labeled training rows for model selection, threshold selection, and final benchmark evaluation.
- Treat public test files as unlabeled scoring/drift inputs only; they SHALL NOT be used for labeled metrics.
- Produce benchmark reports that include schema validation, class balance, split statistics, cost-weighted metrics, and comparison with the current baseline evidence.

## Capabilities

### New Capabilities

- `ieee-cis-benchmark-adapter`: Concrete adapter behavior for preparing, validating, splitting, and reporting on the local IEEE-CIS-style fraud benchmark data.

### Modified Capabilities

- `fraud-benchmark-datasets`: Clarify behavior for benchmark datasets with labeled training data but no public test target labels.

## Impact

- Affected ML areas: dataset registry, benchmark preparation utilities, validation checks, training/evaluation configuration, report generation, and tests.
- Affected docs: benchmark setup and interpretation guidance.
- Raw dataset files remain local and untracked.
- No serving API or cloud provider configuration changes are in scope for this change.
