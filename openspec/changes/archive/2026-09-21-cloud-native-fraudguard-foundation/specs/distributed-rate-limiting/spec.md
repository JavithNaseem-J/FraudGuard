## ADDED Requirements

### Requirement: Distributed rate limiting
The prediction API SHALL use Upstash Redis for distributed rate limiting when Upstash configuration is present.

#### Scenario: Upstash configured
- **WHEN** Upstash REST URL and token are configured
- **THEN** rate limits SHALL be enforced across application instances rather than only within one process

### Requirement: Local rate-limit fallback
The service SHALL preserve a local in-memory rate-limit fallback for development and test environments when Upstash is absent.

#### Scenario: Upstash absent locally
- **WHEN** Upstash configuration is missing
- **THEN** the service SHALL use local rate limiting and clearly identify the fallback mode in non-secret diagnostics

### Requirement: Explicit Redis outage behavior
The service SHALL define and implement explicit behavior for Upstash failures rather than failing unpredictably.

#### Scenario: Upstash request fails
- **WHEN** Redis rate-limit evaluation fails during a prediction request
- **THEN** the service SHALL follow the configured fail-open or fail-closed policy and record a sanitized audit/log event
