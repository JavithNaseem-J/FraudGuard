## MODIFIED Requirements

### Requirement: Drift monitoring reports
The system SHALL provide an executable Evidently workflow for unlabeled feature-summary, prediction-score, and decision drift by comparing a bounded production window with an approved versioned transaction reference profile. Each report SHALL identify model version, artifact release, reference version, evaluated window, row counts, and generated time without containing raw transaction payloads.

#### Scenario: Unlabeled drift report generated
- **WHEN** a sufficient bounded window of production prediction records exists without confirmed labels
- **THEN** the workflow SHALL generate machine-readable metrics and an HTML drift report tied to the selected model and reference versions

#### Scenario: Monitoring window is empty or insufficient
- **WHEN** the selected production window has no records or does not meet the configured minimum sample size
- **THEN** the workflow SHALL exit with an explicit no-data or insufficient-data outcome rather than publishing a misleading drift conclusion

### Requirement: Delayed-label performance reports
The system SHALL provide an executable performance workflow that joins confirmed feedback to prediction records by prediction identifier and reports threshold-dependent metrics, PR-AUC when scores are available, calibration evidence where valid, cost-weighted loss, coverage, model version, and evaluation window.

#### Scenario: Feedback labels available
- **WHEN** confirmed feedback is joinable to predictions in the selected window
- **THEN** the workflow SHALL compute versioned performance metrics without treating unlabeled predictions as negative outcomes

#### Scenario: Labels are not yet available
- **WHEN** no confirmed labels are joinable for the selected window
- **THEN** the workflow SHALL report label coverage and skip supervised performance conclusions

### Requirement: Free-tier observability documentation
The repository SHALL document Evidently execution, report access and retention, Render log usage, Supabase monitoring queries, scheduling options, alert limitations, and free-tier constraints. It SHALL explain that Langfuse is not part of this tabular-ML deployment.

#### Scenario: Operator configures monitoring
- **WHEN** an operator follows the observability documentation
- **THEN** the required commands, approved fields, environment variables, report locations, scheduling options, and free-tier limitations SHALL be clear without exposing secret values or raw transaction records
