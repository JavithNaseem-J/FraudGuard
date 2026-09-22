## 1. Modeling Implementation

- [x] 1.1 Add a transaction benchmark strong-model runner using LightGBM.
- [x] 1.2 Add feature filtering for identifiers, target, high-missing columns, numeric columns, and categorical columns.
- [x] 1.3 Preserve the existing smoke benchmark baseline for comparison.
- [x] 1.4 Keep public unlabeled test data out of metric computation.

## 2. Evaluation and Reporting

- [x] 2.1 Select the operating threshold from validation scores using cost-weighted loss.
- [x] 2.2 Report AP, ROC-AUC, Brier score, precision, recall, F1, confusion matrix, prevalence, and cost-weighted loss.
- [x] 2.3 Compare strong-model results against the numeric smoke baseline in the same report.
- [x] 2.4 Record that serving promotion is not approved by this benchmark alone.

## 3. Professional Naming and Cleanup

- [x] 3.1 Keep user-facing docs/reports on “transaction data” terminology.
- [x] 3.2 Add professional transaction benchmark import aliases.
- [x] 3.3 Remove or bypass bottleneck paths in the benchmark workflow where they block better performance.

## 4. Verification

- [x] 4.1 Add tests for the strong-model benchmark path.
- [x] 4.2 Run unit tests, compile checks, OpenSpec validation, and a bounded real-data benchmark run.
- [x] 4.3 Summarize whether score/performance improved and what remains before serving promotion.
