## ADDED Requirements

### Requirement: Non-serving candidate package
The transaction benchmark SHALL export a reproducible candidate package without replacing serving model artifacts.

#### Scenario: Strong benchmark completes
- **WHEN** the strong transaction benchmark completes successfully
- **THEN** the benchmark output SHALL include a candidate model artifact, threshold metadata, run metadata, and feature audit metadata
- **AND** the report SHALL indicate that serving promotion is not automatic

### Requirement: Promotion gate recording
The transaction benchmark SHALL record explicit promotion gates and whether the candidate passed each gate.

#### Scenario: Candidate metrics evaluated
- **WHEN** candidate metrics are available
- **THEN** the candidate metadata SHALL include gate thresholds, observed metric values, pass/fail status, and an overall candidate decision

### Requirement: Feature schema audit
The candidate package SHALL include a feature schema audit that identifies selected features and schema risks.

#### Scenario: Candidate metadata written
- **WHEN** candidate artifacts are exported
- **THEN** the feature audit SHALL include numeric feature count, categorical feature count, selected feature count, excluded columns, high-missing columns, and whether public unlabeled data was used for metrics

### Requirement: Serving artifact preservation
The candidate export SHALL NOT overwrite the current serving artifact directory.

#### Scenario: Candidate export runs
- **WHEN** candidate artifacts are written
- **THEN** existing serving artifacts SHALL remain untouched unless a separate serving-promotion change explicitly updates them
