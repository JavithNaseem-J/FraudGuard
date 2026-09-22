## MODIFIED Requirements

### Requirement: Dataset registry
The active ML dataset registry SHALL identify the transaction training, optional identity, public test, and optional target artifacts needed by the current benchmark workflow, including target column, schema expectations, and leakage-risk notes. It SHALL NOT require the retired legacy CSV.

#### Scenario: Transaction dataset selected
- **WHEN** the active transaction dataset is selected for validation or training
- **THEN** the pipeline SHALL resolve its configured local artifacts without requiring raw data to be committed to git or the legacy CSV to exist

#### Scenario: Retired dataset name requested
- **WHEN** executable registry code receives the retired legacy dataset identifier
- **THEN** it SHALL reject the request or identify the entry as non-executable historical evidence rather than resolving an active training path

### Requirement: Baseline preservation
Historical baseline metrics and methodology SHALL remain available as clearly labeled comparison evidence, but the retired legacy dataset and its generated artifacts SHALL NOT remain dependencies of active training, serving, CI, or deployment workflows.

#### Scenario: Benchmark comparison is documented
- **WHEN** transaction benchmark results are presented alongside historical baseline results
- **THEN** the report SHALL label the baseline as historical and SHALL NOT imply that its raw data or pipeline is required for production

#### Scenario: Active pipeline executes
- **WHEN** validation, training, testing, container build, or production startup runs
- **THEN** it SHALL complete without reading or requiring the retired legacy CSV
