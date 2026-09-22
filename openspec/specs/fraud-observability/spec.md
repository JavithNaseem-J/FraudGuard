# fraud-observability Specification

## Purpose
TBD - created by archiving change cloud-native-fraudguard-foundation. Update Purpose after archive.
## Requirements
### Requirement: Structured service telemetry
The service SHALL emit structured logs or metrics that include request ID, prediction ID, model version, model mode, latency, status, row count where applicable, security outcome, and sanitized error category without logging provider secrets or raw sensitive payloads.

#### Scenario: Prediction request logged
- **WHEN** a prediction request completes
- **THEN** telemetry SHALL include operational metadata sufficient for debugging and dashboarding without exposing raw transaction details

#### Scenario: Transaction batch request logged
- **WHEN** a transaction candidate batch request completes
- **THEN** telemetry SHALL include model mode, row count, success/failure status, latency, and sanitized validation error category

#### Scenario: Protected request denied
- **WHEN** a protected endpoint denies access
- **THEN** telemetry SHALL include a sanitized security outcome without recording the provided API key or raw payload

### Requirement: Drift monitoring reports
The system SHALL support Evidently OSS report generation for unlabeled input, feature, and prediction drift by comparing production prediction records with an approved reference dataset. For transaction candidate serving, drift inputs SHALL be compatible with the candidate feature schema or an explicitly documented summary schema.

#### Scenario: Unlabeled drift report generated
- **WHEN** prediction records exist without confirmed labels
- **THEN** the monitoring workflow SHALL be able to generate a drift report using reference and current prediction data

#### Scenario: Candidate drift report generated
- **WHEN** transaction candidate prediction records exist
- **THEN** the monitoring workflow SHALL be able to compare candidate score and feature summaries against the approved transaction reference data

### Requirement: Delayed-label performance reports
The system SHALL support model performance reports after feedback labels are available and joinable to prediction records.

#### Scenario: Feedback labels available
- **WHEN** prediction feedback contains confirmed labels
- **THEN** the monitoring workflow SHALL compute performance metrics for the corresponding prediction window

### Requirement: Free-tier observability documentation
The repository SHALL document how to connect service telemetry and ML reports to free-friendly observability tools and SHALL state limitations of those free tiers.

#### Scenario: Operator follows setup docs
- **WHEN** an operator configures Grafana Cloud and Evidently report generation from the docs
- **THEN** the required environment variables, commands, storage locations, and limitations SHALL be clear without exposing secret values
