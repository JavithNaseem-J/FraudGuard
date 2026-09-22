## MODIFIED Requirements

### Requirement: Deployment preflight checks
The repository SHALL include deployment preflight checks that validate security settings, request limits, provider configuration, immutable release selection, artifact integrity, transaction-model compatibility, and rollback metadata without printing secret values. Production transaction preflight SHALL NOT require legacy baseline artifacts.

#### Scenario: Transaction production preflight succeeds
- **WHEN** authentication, limits, Supabase, Upstash, release selection, verified transaction artifacts, and rollback metadata meet deployment requirements
- **THEN** preflight SHALL succeed and emit only sanitized configuration-presence and release metadata

#### Scenario: Required production configuration is missing
- **WHEN** a required security, provider, artifact, or release setting is absent or invalid
- **THEN** preflight SHALL fail before deployment and identify the non-secret corrective action

#### Scenario: Legacy baseline artifacts are absent
- **WHEN** production transaction preflight runs from a clean checkout without baseline artifacts
- **THEN** their absence SHALL NOT be reported as a failure

### Requirement: Deployment documentation and safe claims
The repository SHALL document clean-checkout Render deployment, private Supabase artifact publication, database migrations, Upstash setup, Evidently monitoring, verified metrics, smoke checks, rollback, and free-tier limitations using professional transaction-data wording.

#### Scenario: Operator follows production documentation
- **WHEN** an operator follows the documented release procedure
- **THEN** required accounts, protected variables, commands, migrations, artifact expectations, validation evidence, monitoring setup, rollback steps, and limitations SHALL be clear without exposing secrets

#### Scenario: Production readiness is described
- **WHEN** project documentation makes a production-related claim
- **THEN** the claim SHALL distinguish implemented code, locally verified behavior, externally deployed evidence, and free-tier limitations
