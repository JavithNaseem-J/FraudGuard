## MODIFIED Requirements

### Requirement: No automatic serving promotion
The benchmark SHALL NOT replace the deployed serving model automatically, even when it writes a non-serving candidate package.

#### Scenario: Strong candidate outperforms baseline
- **WHEN** a strong candidate outperforms the smoke baseline
- **THEN** the report SHALL mark promotion as a separate decision rather than updating serving artifacts
- **AND** any exported model package SHALL be marked as a benchmark candidate, not the active serving model
