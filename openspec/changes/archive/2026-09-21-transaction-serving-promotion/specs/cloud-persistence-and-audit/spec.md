## MODIFIED Requirements

### Requirement: Supabase prediction persistence
The system SHALL persist successful prediction requests to Supabase when configured, including request identifier, model version, score, threshold, decision, calibration flag, model mode, and created timestamp.

#### Scenario: Prediction persisted
- **WHEN** a prediction completes and Supabase is configured
- **THEN** a prediction record SHALL be written with enough metadata to reproduce which model release and operating threshold produced the decision

#### Scenario: Supabase unavailable
- **WHEN** Supabase is not configured in local development
- **THEN** prediction serving SHALL continue with explicit non-persistent fallback behavior

#### Scenario: Batch candidate predictions persisted
- **WHEN** a transaction candidate batch prediction completes and Supabase is configured
- **THEN** each scored row SHALL be joinable to its prediction ID and model release metadata

### Requirement: Model release registry
The system SHALL store model release metadata in Supabase when configured, including model version, artifact identifiers, threshold, cost assumptions, calibration flag, feature schema summary, model mode, and release timestamp.

#### Scenario: Model release recorded
- **WHEN** a model artifact is promoted for serving
- **THEN** release metadata SHALL be persisted without storing raw secret values or raw training data

#### Scenario: Transaction candidate release recorded
- **WHEN** a transaction candidate is enabled for serving
- **THEN** release metadata SHALL include candidate artifact identifiers, threshold metadata, feature count, and promotion evidence summary
