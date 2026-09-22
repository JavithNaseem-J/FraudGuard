## MODIFIED Requirements

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

### Requirement: Render-compatible deployment configuration
The repository SHALL include Render-compatible deploy configuration and runtime documentation that uses provider-injected ports and environment variables.

#### Scenario: Render reads deployment config
- **WHEN** the repository is connected to Render
- **THEN** the deployment configuration SHALL define the web service build/start behavior and required environment variables without embedding secret values

#### Scenario: Candidate deployment configured
- **WHEN** transaction candidate mode is configured on Render
- **THEN** documented environment variables SHALL identify model mode and candidate artifact location without committing provider secrets
