## MODIFIED Requirements

### Requirement: Distributed rate limiting
The prediction API SHALL use Upstash Redis for distributed rate limiting when Upstash configuration is present. Rate limiting SHALL work alongside API authentication and request-size guardrails.

#### Scenario: Upstash configured
- **WHEN** Upstash REST URL and token are configured
- **THEN** rate limits SHALL be enforced across application instances rather than only within one process

#### Scenario: Authenticated request is rate limited
- **WHEN** a protected prediction request passes authentication
- **THEN** rate limiting SHALL still be evaluated before prediction work executes

### Requirement: Local rate-limit fallback
The service SHALL preserve a local in-memory rate-limit fallback for development and test environments when Upstash is absent.

#### Scenario: Upstash absent locally
- **WHEN** Upstash configuration is missing
- **THEN** the service SHALL use local rate limiting and clearly identify the fallback mode in non-secret diagnostics

#### Scenario: Local authenticated request is limited
- **WHEN** local rate limiting is active
- **THEN** protected endpoints SHALL still enforce request limits after authentication succeeds
