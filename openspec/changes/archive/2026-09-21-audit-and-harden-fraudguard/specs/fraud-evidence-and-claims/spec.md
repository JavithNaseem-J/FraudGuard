## ADDED Requirements

### Requirement: Executed verification evidence
The project SHALL distinguish checks that passed by execution, checks that
failed, and checks not executed or requiring a full run. Generated metrics SHALL
not be described as verified unless their producing pipeline ran successfully.

#### Scenario: Partial local verification
- **WHEN** unit and smoke tests run but full model training cannot run
- **THEN** documentation identifies the passed checks and lists full retraining/evaluation as not executed

### Requirement: Reproducible pipeline contract
Configuration and DVC SHALL declare the actual dependencies and outputs of
ingestion, validation, split preparation, training, evaluation, and serving.

#### Scenario: DVC dependency graph
- **WHEN** a source, configuration, split, or model input changes
- **THEN** DVC identifies every downstream stage whose outputs require regeneration without relying on undeclared legacy artifacts

### Requirement: Defensible project claims
Documentation and resume/interview guidance SHALL describe only behavior and
metrics supported by the repository or reproducible evidence, and SHALL state
the split, calibration, deployment, and dataset limitations.

#### Scenario: Report model quality
- **WHEN** project documentation mentions model quality
- **THEN** it cites the evaluation protocol and verified metric artifact or clearly labels the number as historical/unverified

#### Scenario: Describe production scope
- **WHEN** the project is presented as real-time or production-oriented
- **THEN** the claim is limited to implemented API behavior and does not imply temporal validation, calibrated financial risk, or production monitoring without evidence
