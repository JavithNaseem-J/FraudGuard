# transaction-benchmark-modeling Specification

## Purpose
TBD - created by archiving change transaction-data-full-benchmark-model. Update Purpose after archive.
## Requirements
### Requirement: Strong tabular benchmark candidate
The system SHALL support a stronger transaction-data benchmark candidate using LightGBM when LightGBM is installed.

#### Scenario: LightGBM available
- **WHEN** the strong benchmark runner is executed and LightGBM is installed
- **THEN** it SHALL train a LightGBM candidate on labeled transaction training rows and report evaluation metrics on the internal labeled benchmark test split

### Requirement: Baseline comparison
The benchmark report SHALL compare the stronger candidate against the numeric logistic smoke baseline.

#### Scenario: Benchmark report generated
- **WHEN** the strong benchmark completes
- **THEN** the report SHALL include both baseline and strong-candidate metrics with a clear best-candidate indicator

### Requirement: Cost-weighted thresholding
The strong benchmark SHALL select its operating threshold from validation scores using configured false-positive and false-negative costs.

#### Scenario: Threshold selected
- **WHEN** validation predictions are available
- **THEN** the benchmark SHALL select the threshold using cost-weighted loss and SHALL report the cost assumptions

### Requirement: No public test metrics
The strong benchmark SHALL NOT use public unlabeled test data for precision, recall, AP, ROC-AUC, F1, Brier score, confusion matrix, or cost-weighted metrics.

#### Scenario: Public test files present
- **WHEN** public unlabeled test files exist locally
- **THEN** the benchmark SHALL still compute labeled metrics only from internal labeled splits

### Requirement: No automatic serving promotion
The benchmark SHALL NOT replace the deployed serving model automatically, even when it writes a non-serving candidate package.

#### Scenario: Strong candidate outperforms baseline
- **WHEN** a strong candidate outperforms the smoke baseline
- **THEN** the report SHALL mark promotion as a separate decision rather than updating serving artifacts
- **AND** any exported model package SHALL be marked as a benchmark candidate, not the active serving model
