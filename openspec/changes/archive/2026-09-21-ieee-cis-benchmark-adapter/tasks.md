## 1. Dataset Contract

- [x] 1.1 Document the exact local IEEE-CIS file contract and confirm raw files remain untracked.
- [x] 1.2 Add adapter configuration for transaction, identity, target, join key, split seed, sample mode, and output paths.
- [x] 1.3 Add validation for required columns, `isFraud` binary target, join key presence, class balance, duplicate transaction IDs, and missing identity coverage.

## 2. Preparation and Splitting

- [x] 2.1 Implement a preparation utility that reads `train_transaction.csv` and optionally left-joins `train_identity.csv`.
- [x] 2.2 Preserve all labeled transaction rows when identity data is missing.
- [x] 2.3 Create deterministic labeled train/validation/test partitions from `train_transaction.csv`.
- [x] 2.4 Ensure public/unlabeled test files are never used for labeled benchmark metrics.

## 3. Benchmark Training and Evaluation

- [x] 3.1 Produce prepared benchmark artifacts under an ignored artifact directory.
- [x] 3.2 Run a smoke benchmark on a bounded sample to verify memory/runtime behavior.
- [x] 3.3 Run cost-weighted threshold selection using training-only validation or out-of-fold scores.
- [x] 3.4 Evaluate the final benchmark split with AP, ROC-AUC, Brier score, precision, recall, F1, confusion matrix, prevalence, and cost-weighted loss.
- [x] 3.5 Write a benchmark report comparing current baseline evidence and IEEE-CIS benchmark evidence without claiming serving promotion.

## 4. Tests and Documentation

- [x] 4.1 Add tests for path validation, identity left join behavior, deterministic splitting, and unlabeled-public-test handling.
- [x] 4.2 Add documentation for running smoke and full benchmark modes.
- [x] 4.3 Run OpenSpec validation, unit tests, and at least one smoke preparation check.
