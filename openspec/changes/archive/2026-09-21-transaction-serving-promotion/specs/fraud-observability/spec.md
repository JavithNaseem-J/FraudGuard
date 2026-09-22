## MODIFIED Requirements

### Requirement: Structured service telemetry
The service SHALL emit structured logs or metrics that include request ID, prediction ID, model version, model mode, latency, status, row count where applicable, and sanitized error category without logging provider secrets or raw sensitive payloads.

#### Scenario: Prediction request logged
- **WHEN** a prediction request completes
- **THEN** telemetry SHALL include operational metadata sufficient for debugging and dashboarding without exposing raw transaction details

#### Scenario: Transaction batch request logged
- **WHEN** a transaction candidate batch request completes
- **THEN** telemetry SHALL include model mode, row count, success/failure status, latency, and sanitized validation error category

### Requirement: Drift monitoring reports
The system SHALL support Evidently OSS report generation for unlabeled input, feature, and prediction drift by comparing production prediction records with an approved reference dataset. For transaction candidate serving, drift inputs SHALL be compatible with the candidate feature schema or an explicitly documented summary schema.

#### Scenario: Unlabeled drift report generated
- **WHEN** prediction records exist without confirmed labels
- **THEN** the monitoring workflow SHALL be able to generate a drift report using reference and current prediction data

#### Scenario: Candidate drift report generated
- **WHEN** transaction candidate prediction records exist
- **THEN** the monitoring workflow SHALL be able to compare candidate score and feature summaries against the approved transaction reference data
