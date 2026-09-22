## MODIFIED Requirements

### Requirement: Provider split preservation
When a benchmark dataset is delivered with separate training, test, and target artifacts, the pipeline SHALL preserve that split and SHALL NOT randomly combine and re-split provider test rows into training data. When a benchmark dataset has labeled training data but no public test target labels, the pipeline SHALL create metrics only from deterministic internal splits of the labeled training data and SHALL reserve public test rows for unlabeled scoring or drift workflows.

#### Scenario: Pre-split benchmark dataset evaluated
- **WHEN** a registered benchmark dataset provides separate training, test, and target artifacts
- **THEN** model selection SHALL use only the training artifact and final benchmark evaluation SHALL use the provider test artifact joined to its target labels when labels are available

#### Scenario: Public test labels unavailable
- **WHEN** a registered benchmark dataset provides labeled training data but no public test target labels
- **THEN** benchmark metrics SHALL use deterministic internal splits from the labeled training data and SHALL NOT report labeled metrics for the public test rows
