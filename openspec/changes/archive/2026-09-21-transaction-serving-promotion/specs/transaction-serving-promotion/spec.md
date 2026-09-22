## ADDED Requirements

### Requirement: Feature-flagged transaction candidate serving
The service SHALL serve the transaction-data candidate only when explicitly enabled by runtime configuration.

#### Scenario: Baseline mode startup
- **WHEN** candidate serving is not enabled
- **THEN** the service SHALL keep the current serving path active and SHALL NOT require transaction candidate artifacts

#### Scenario: Candidate mode startup
- **WHEN** candidate serving is enabled
- **THEN** the service SHALL load and validate the candidate model, threshold, metadata, and feature audit before reporting ready

### Requirement: Batch transaction prediction contract
The service SHALL provide a batch transaction prediction contract for rows matching the candidate feature schema.

#### Scenario: Valid transaction batch submitted
- **WHEN** one or more valid transaction rows are submitted to the candidate endpoint
- **THEN** the service SHALL return one prediction result per row with prediction ID, score, threshold, decision, model metadata, and validation status

#### Scenario: Required feature missing
- **WHEN** a submitted transaction row omits a required candidate feature
- **THEN** the service SHALL reject that row or request with a validation error naming the missing feature

### Requirement: Candidate fallback safety
The candidate serving path SHALL NOT remove or silently alter the existing manual prediction path.

#### Scenario: Candidate unavailable
- **WHEN** candidate mode is disabled or candidate artifacts are unavailable in baseline mode
- **THEN** the existing manual prediction endpoint SHALL continue to use the baseline serving artifact behavior

### Requirement: Candidate deployment preflight
The repository SHALL include a preflight check for transaction candidate serving configuration.

#### Scenario: Predeploy check runs
- **WHEN** the preflight check is executed for candidate mode
- **THEN** it SHALL verify required environment variables and candidate artifact files without exposing secret values
