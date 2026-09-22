## ADDED Requirements

### Requirement: Supabase prediction persistence
The system SHALL persist successful prediction requests to Supabase when configured, including request identifier, model version, score, threshold, decision, calibration flag, and created timestamp.

#### Scenario: Prediction persisted
- **WHEN** a prediction completes and Supabase is configured
- **THEN** a prediction record SHALL be written with enough metadata to reproduce which model release and operating threshold produced the decision

#### Scenario: Supabase unavailable
- **WHEN** Supabase is not configured in local development
- **THEN** prediction serving SHALL continue with explicit non-persistent fallback behavior

### Requirement: Prediction feedback capture
The system SHALL support delayed feedback records linked to prediction identifiers so later labels or reviewer corrections can be joined to prior predictions.

#### Scenario: Feedback submitted for prediction
- **WHEN** feedback is submitted with a known prediction ID
- **THEN** the system SHALL store the feedback and make it joinable to the original prediction record

### Requirement: Model release registry
The system SHALL store model release metadata in Supabase when configured, including model version, artifact identifiers, threshold, cost assumptions, calibration flag, and release timestamp.

#### Scenario: Model release recorded
- **WHEN** a model artifact is promoted for serving
- **THEN** release metadata SHALL be persisted without storing raw secret values or raw training data

### Requirement: Audit event history
The system SHALL record audit events for material serving actions such as prediction creation, feedback creation, model release registration, and provider integration failures.

#### Scenario: Auditable event occurs
- **WHEN** a material serving action occurs
- **THEN** an audit event SHALL be written with timestamp, event type, relevant identifiers, and sanitized metadata

### Requirement: Server-only Supabase credentials
Supabase service-role credentials SHALL be used only by the server process and SHALL NOT be exposed to templates, browser JavaScript, logs, or committed files.

#### Scenario: Rendering a user-facing page
- **WHEN** the application renders any browser-facing response
- **THEN** no Supabase service-role key or provider secret SHALL be included
