## ADDED Requirements

### Requirement: Local IEEE-CIS file contract
The adapter SHALL define a local file contract for IEEE-CIS-style data that uses `train_transaction.csv` as the labeled source, `train_identity.csv` as optional training identity features, and `test_identity.csv` or other public test files only as unlabeled scoring inputs.

#### Scenario: File contract validated
- **WHEN** the adapter validates local IEEE-CIS data
- **THEN** it SHALL require the labeled training transaction file and SHALL NOT require public test target labels

### Requirement: Labeled training source
The adapter SHALL treat `isFraud` in the labeled training transaction file as the positive-class target and SHALL validate that it is binary before benchmark training.

#### Scenario: Target validated
- **WHEN** the labeled training transaction file is loaded
- **THEN** the adapter SHALL confirm `isFraud` exists and contains binary labels before producing training artifacts

### Requirement: Optional identity left join
The adapter SHALL optionally left-join identity features by `TransactionID` and SHALL preserve transaction rows that have no matching identity row.

#### Scenario: Missing identity rows
- **WHEN** a labeled transaction has no matching identity row
- **THEN** the prepared dataset SHALL retain the transaction with missing identity feature values

### Requirement: Deterministic internal benchmark split
The adapter SHALL create deterministic labeled train, validation or threshold-selection, and final benchmark test partitions from the labeled training source.

#### Scenario: Repeat split
- **WHEN** the same labeled source and split seed are used twice
- **THEN** the adapter SHALL produce identical partition membership and class counts

### Requirement: Unlabeled public test separation
The adapter SHALL NOT use public unlabeled test rows to compute benchmark metrics.

#### Scenario: Benchmark metrics generated
- **WHEN** benchmark metrics are computed
- **THEN** the metric rows SHALL come only from labeled internal validation or benchmark test partitions

### Requirement: Benchmark report
The adapter SHALL produce a benchmark report that includes data validation status, split sizes, prevalence, selected threshold policy, ranking metrics, decision metrics, and cost-weighted loss.

#### Scenario: Report written
- **WHEN** benchmark evaluation completes
- **THEN** the report SHALL include enough metadata to distinguish IEEE-CIS benchmark results from the deployed serving model
