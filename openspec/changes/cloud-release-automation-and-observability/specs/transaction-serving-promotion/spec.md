## MODIFIED Requirements

### Requirement: Feature-flagged transaction candidate serving
The service SHALL support explicit local mode selection and SHALL use the promoted transaction model as the primary production serving mode. Production transaction startup SHALL NOT initialize the legacy baseline predictor.

#### Scenario: Local baseline mode explicitly selected
- **WHEN** a developer explicitly selects the documented local legacy mode while legacy artifacts remain available
- **THEN** the service MAY expose the historical manual path without affecting transaction model artifacts or production defaults

#### Scenario: Production transaction mode selected
- **WHEN** the application runs in Render, staging, or production with the transaction release configured
- **THEN** it SHALL load and validate only the transaction release required by the production prediction contract

### Requirement: Candidate deployment preflight
The repository SHALL include a preflight check for production transaction serving that validates security settings, provider configuration, release selection, manifest integrity, and candidate compatibility without requiring baseline artifacts.

#### Scenario: Production predeploy check runs
- **WHEN** the preflight check is executed for transaction mode
- **THEN** it SHALL verify required environment configuration and the selected transaction release while omitting secret values

#### Scenario: Baseline artifacts are absent
- **WHEN** transaction-mode preflight runs without legacy baseline artifacts
- **THEN** preflight SHALL succeed if the selected transaction release and all production requirements are valid

## REMOVED Requirements

### Requirement: Candidate fallback safety
**Reason**: Keeping the legacy manual predictor as an implicit fallback makes production readiness depend on artifacts produced from the retired legacy dataset and can silently change the prediction contract.

**Migration**: Use the transaction batch prediction contract for production. Historical baseline behavior, if retained temporarily, must be explicitly selected in local mode and must not participate in production startup or readiness.
