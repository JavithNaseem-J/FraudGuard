## MODIFIED Requirements

### Requirement: Startup artifact loading and readiness
The service SHALL bootstrap, verify, load, and validate the configured transaction model release during startup or application lifespan initialization and SHALL expose readiness based on that selected release. In production transaction mode, startup and readiness SHALL NOT require legacy baseline artifacts or the legacy CSV.

#### Scenario: Remote transaction release loads successfully
- **WHEN** the configured release is downloaded or found in a matching verified cache and all model, threshold, metadata, feature-audit, and manifest checks pass
- **THEN** readiness SHALL report ready with sanitized release identifier, model version, threshold, and model-mode metadata

#### Scenario: Selected release is unavailable or invalid
- **WHEN** download, integrity, schema, compatibility, or model loading fails
- **THEN** liveness SHALL remain available, readiness SHALL fail, and diagnostics SHALL omit filesystem internals, credentials, signed URLs, and stack traces

#### Scenario: Production transaction mode selected
- **WHEN** runtime configuration selects the production transaction model
- **THEN** the service SHALL initialize that model independently and SHALL NOT load or validate the legacy baseline model as a readiness dependency

### Requirement: Render-compatible deployment configuration
The repository SHALL include Render-compatible deployment configuration and runtime documentation using provider-injected ports, protected environment variables, transaction-model mode, and an immutable artifact release identifier.

#### Scenario: Render reads deployment config
- **WHEN** the repository is connected to Render
- **THEN** the deployment configuration SHALL define the container runtime, health path, transaction model mode, release selection, limits, and required secret names without embedding secret values

#### Scenario: Clean Render checkout starts
- **WHEN** Render builds from a clean repository checkout with configured provider secrets and a published model release
- **THEN** the service SHALL obtain its verified production artifacts without requiring git-ignored local files

#### Scenario: Render security variables configured
- **WHEN** the service is deployed publicly
- **THEN** deployment documentation SHALL require API authentication, request guardrails, provider credentials, and release selection to be reviewed before public use
