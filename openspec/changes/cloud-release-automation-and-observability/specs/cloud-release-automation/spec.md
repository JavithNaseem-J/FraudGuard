## ADDED Requirements

### Requirement: Reproducible clean-checkout CI
The continuous-integration workflow SHALL install the project package and run critical checks from a clean checkout without depending on a manually configured local import path or untracked production artifacts.

#### Scenario: Pull request CI runs
- **WHEN** a pull request or protected-branch push triggers CI
- **THEN** package installation, tests, OpenSpec validation, import or compilation checks, and deployment preflight tests SHALL run with deterministic dependencies

### Requirement: Gated Render deployment
The delivery workflow SHALL deploy to Render only after required CI gates pass and SHALL use protected deployment credentials without retaining the legacy AWS ECR or self-hosted-runner dependency.

#### Scenario: CI gate fails
- **WHEN** any required verification step fails
- **THEN** the workflow SHALL NOT trigger a Render deployment

#### Scenario: CI gates pass
- **WHEN** all required checks pass on the protected deployment branch
- **THEN** the workflow SHALL trigger the configured Render deployment without exposing its deploy credential

### Requirement: Container and post-deployment smoke verification
The release workflow SHALL verify the built container locally and the deployed Render service remotely before treating the release as successful.

#### Scenario: Container smoke test runs
- **WHEN** the production container image is built in CI
- **THEN** the workflow SHALL verify process liveness, model readiness, authentication enforcement, and a non-sensitive fixture prediction

#### Scenario: Render deployment becomes ready
- **WHEN** Render reports a deployed revision
- **THEN** the workflow SHALL poll liveness and readiness and execute an authenticated transaction prediction using protected test credentials

#### Scenario: Deployed smoke test fails
- **WHEN** readiness or the authenticated transaction prediction fails within the bounded verification window
- **THEN** the workflow SHALL mark the release failed and provide a documented rollback command or release-selection procedure

### Requirement: Secret-safe provider configuration
CI and deployment automation SHALL read Supabase, Upstash, Render, and application credentials only from protected secret stores and SHALL NOT print their values.

#### Scenario: Workflow diagnostics are retained
- **WHEN** CI or deployment logs are reviewed
- **THEN** they SHALL show configuration presence and sanitized outcomes without provider tokens, service-role keys, API keys, deploy hooks, or signed artifact URLs
