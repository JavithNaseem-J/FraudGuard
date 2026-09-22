## MODIFIED Requirements

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
