## MODIFIED Requirements

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
