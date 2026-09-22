## ADDED Requirements

### Requirement: Dataset registry
The ML pipeline SHALL support a dataset registry that identifies the current baseline dataset and at least one external benchmark dataset by name, local training path, local test path, local target path when separate labels are provided, target column, schema expectations, and leakage-risk notes.

#### Scenario: Registered dataset selected
- **WHEN** a dataset name is selected for validation or training
- **THEN** the pipeline SHALL resolve its configured training, test, and target artifacts without requiring raw data to be committed to git

### Requirement: Provider split preservation
When a benchmark dataset is delivered with separate training, test, and target artifacts, the pipeline SHALL preserve that split and SHALL NOT randomly combine and re-split provider test rows into training data.

#### Scenario: Pre-split benchmark dataset evaluated
- **WHEN** a registered benchmark dataset provides separate training, test, and target artifacts
- **THEN** model selection SHALL use only the training artifact and final benchmark evaluation SHALL use the provider test artifact joined to its target labels when labels are available

### Requirement: Baseline preservation
The current dataset SHALL remain available as a baseline so benchmark metrics can be compared against the existing project evidence.

#### Scenario: Benchmark evaluation runs
- **WHEN** an external benchmark dataset is evaluated
- **THEN** the report SHALL identify the baseline dataset and benchmark dataset separately rather than overwriting baseline evidence

### Requirement: Benchmark dataset validation
The pipeline SHALL validate registered benchmark datasets for target availability or joinability, train/test split integrity, class balance, duplicate records, required feature availability at prediction time, and known leakage-risk columns before training or evaluation.

#### Scenario: Leakage-risk column detected
- **WHEN** a registered dataset contains columns documented as post-event, target-derived, or otherwise unavailable at prediction time
- **THEN** validation SHALL fail or require an explicit exclusion before model training proceeds

### Requirement: External raw data exclusion
Raw external benchmark datasets SHALL NOT be committed to the repository.

#### Scenario: Documentation references benchmark data
- **WHEN** setup documentation describes a benchmark dataset
- **THEN** it SHALL provide local placement and checksum guidance without adding the raw dataset file to version control
