# api-security-and-deploy-hardening Specification

## Purpose
TBD - created by archiving change production-deploy-and-security-hardening. Update Purpose after archive.
## Requirements
### Requirement: Protected prediction endpoints
The service SHALL support API-key protection for prediction and feedback endpoints while preserving public health checks.

#### Scenario: Protected request missing API key
- **WHEN** API authentication is required and a protected endpoint receives no API key
- **THEN** the service SHALL reject the request without running prediction or persistence logic

#### Scenario: Protected request includes valid API key
- **WHEN** API authentication is required and a protected endpoint receives the configured API key
- **THEN** the service SHALL process the request normally

#### Scenario: Readiness checked without API key
- **WHEN** a caller requests liveness or readiness
- **THEN** the service SHALL respond without requiring an API key

### Requirement: Request guardrails
The service SHALL enforce configurable request-size and batch-size limits before expensive prediction work.

#### Scenario: Request body exceeds limit
- **WHEN** a request body exceeds the configured byte limit
- **THEN** the service SHALL reject it before prediction execution

#### Scenario: Batch row count exceeds limit
- **WHEN** a transaction batch exceeds the configured row limit
- **THEN** the service SHALL reject it with a clear validation error

### Requirement: Deployment preflight checks
The repository SHALL include a deployment preflight check that validates required runtime settings without printing secret values.

#### Scenario: Preflight runs in baseline mode
- **WHEN** the preflight command is run for baseline mode
- **THEN** it SHALL validate baseline artifact paths and security/deployment settings that apply to baseline serving

#### Scenario: Preflight runs in candidate mode
- **WHEN** the preflight command is run for transaction candidate mode
- **THEN** it SHALL also validate candidate artifacts and candidate runtime settings

### Requirement: Deployment documentation and safe claims
The repository SHALL document deployment steps, artifact delivery options, verified metrics, and limitations using professional transaction-data wording.

#### Scenario: Operator follows documentation
- **WHEN** an operator follows the README and cloud deployment docs
- **THEN** required environment variables, commands, migrations, artifact expectations, and limitations SHALL be clear without exposing secrets
