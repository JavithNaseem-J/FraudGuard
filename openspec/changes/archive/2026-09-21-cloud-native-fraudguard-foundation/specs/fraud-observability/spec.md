## ADDED Requirements

### Requirement: Structured service telemetry
The service SHALL emit structured logs or metrics that include request ID, prediction ID, model version, latency, status, and sanitized error category without logging provider secrets or raw sensitive payloads.

#### Scenario: Prediction request logged
- **WHEN** a prediction request completes
- **THEN** telemetry SHALL include operational metadata sufficient for debugging and dashboarding without exposing raw transaction details

### Requirement: Drift monitoring reports
The system SHALL support Evidently OSS report generation for unlabeled input, feature, and prediction drift by comparing production prediction records with an approved reference dataset.

#### Scenario: Unlabeled drift report generated
- **WHEN** prediction records exist without confirmed labels
- **THEN** the monitoring workflow SHALL be able to generate a drift report using reference and current prediction data

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
