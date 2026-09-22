# cloud-service-runtime Specification

## Purpose
TBD - created by archiving change cloud-native-fraudguard-foundation. Update Purpose after archive.
## Requirements
### Requirement: Environment-driven cloud runtime configuration
The service SHALL read cloud runtime settings from environment variables with safe local defaults and SHALL never require provider secrets to be present for local development. Security-related settings SHALL also be environment-driven and SHALL NOT log configured secret values.

#### Scenario: Local startup without cloud secrets
- **WHEN** the application starts without Supabase or Upstash environment variables
- **THEN** it SHALL start in local mode with explicit fallback behavior and no secret-related crash

#### Scenario: Cloud startup with provider settings
- **WHEN** provider environment variables are present
- **THEN** the service SHALL use them for cloud integrations without logging their values

#### Scenario: API auth configured
- **WHEN** API authentication environment variables are configured
- **THEN** protected endpoints SHALL enforce them without exposing the configured key

### Requirement: Startup artifact loading and readiness
The service SHALL load and validate model artifacts during startup or application lifespan initialization and SHALL expose readiness based on the loaded artifact state. Readiness SHALL distinguish baseline mode from transaction candidate mode.

#### Scenario: Artifacts load successfully
- **WHEN** all required model, threshold, and model metadata artifacts are present and compatible
- **THEN** the readiness endpoint SHALL report ready with model version and threshold metadata

#### Scenario: Artifacts are unavailable
- **WHEN** required artifacts are missing or incompatible
- **THEN** the readiness endpoint SHALL report not ready without exposing filesystem internals or stack traces

#### Scenario: Candidate mode selected
- **WHEN** runtime configuration selects transaction candidate mode
- **THEN** readiness SHALL include candidate artifact status, model mode, and sanitized model metadata

### Requirement: Separate liveness and readiness checks
The service SHALL provide separate liveness and readiness checks for cloud deployment health monitoring.

#### Scenario: Process is alive but model is unavailable
- **WHEN** the web process is running but model artifacts are not ready
- **THEN** liveness SHALL succeed and readiness SHALL fail

### Requirement: Safe prediction API response contract
The prediction API SHALL return a structured JSON response that includes a server-generated prediction identifier, score, threshold, decision, model version, and calibration flag, and SHALL NOT place transaction payloads in URL query strings.

#### Scenario: Prediction succeeds
- **WHEN** a valid transaction request is submitted
- **THEN** the response SHALL include prediction metadata and a prediction ID without requiring sensitive request data in a redirect URL

### Requirement: Render-compatible deployment configuration
The repository SHALL include Render-compatible deploy configuration and runtime documentation that uses provider-injected ports and environment variables.

#### Scenario: Render reads deployment config
- **WHEN** the repository is connected to Render
- **THEN** the deployment configuration SHALL define the web service build/start behavior and required environment variables without embedding secret values

#### Scenario: Candidate deployment configured
- **WHEN** transaction candidate mode is configured on Render
- **THEN** documented environment variables SHALL identify model mode and candidate artifact location without committing provider secrets

#### Scenario: Render security variables configured
- **WHEN** the service is deployed publicly
- **THEN** deployment documentation SHALL require API authentication and request guardrail variables to be reviewed before public use
