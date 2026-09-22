## MODIFIED Requirements

### Requirement: Audit event history
The system SHALL record audit events for material serving actions such as prediction creation, feedback creation, model release registration, provider integration failures, denied protected requests, and deployment preflight failures.

#### Scenario: Auditable event occurs
- **WHEN** a material serving action occurs
- **THEN** an audit event SHALL be written with timestamp, event type, relevant identifiers, and sanitized metadata

#### Scenario: Protected request denied
- **WHEN** a protected endpoint rejects a request for missing or invalid credentials
- **THEN** the denial SHALL be logged or audited with sanitized metadata and without storing the submitted API key

### Requirement: Server-only Supabase credentials
Supabase service-role credentials SHALL be used only by the server process and SHALL NOT be exposed to templates, browser JavaScript, logs, committed files, preflight output, or documentation examples.

#### Scenario: Rendering a user-facing page
- **WHEN** the application renders any browser-facing response
- **THEN** no Supabase service-role key or provider secret SHALL be included

#### Scenario: Deployment diagnostics printed
- **WHEN** deployment preflight diagnostics are printed
- **THEN** provider secret values SHALL be redacted or omitted
