## MODIFIED Requirements

### Requirement: Serving artifact preservation
The candidate export SHALL NOT overwrite the current serving artifact directory unless an explicit serving-promotion flow validates the candidate serving contract and runtime configuration.

#### Scenario: Candidate export runs
- **WHEN** candidate artifacts are written
- **THEN** existing serving artifacts SHALL remain untouched unless a separate serving-promotion change explicitly updates them

#### Scenario: Candidate enters serving flow
- **WHEN** a candidate is selected for serving promotion
- **THEN** the promotion flow SHALL validate artifact compatibility, runtime mode, feature schema, persistence metadata, and readiness behavior before the candidate is used for live predictions
