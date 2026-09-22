## MODIFIED Requirements

### Requirement: Imbalance-aware model selection
The system SHALL compare a simple baseline and candidate models using metrics
appropriate to imbalanced fraud ranking and cost-aware operation, and SHALL
apply imbalance handling without resampling validation or test observations.

#### Scenario: Candidate selection
- **WHEN** model selection completes
- **THEN** the selected model SHALL be supported by training-only cross-validation average precision, cost-weighted evaluation using configured costs, and comparison with a recorded baseline

### Requirement: Training-only threshold selection
The decision threshold SHALL be selected from training-only validation or
out-of-fold scores, with its objective, false-positive cost, false-negative
cost, and value stored as model metadata. The final test set SHALL not influence
the threshold.

#### Scenario: Select operating threshold
- **WHEN** a model family and hyperparameters have been selected
- **THEN** the system SHALL derive the operating threshold from training-only scores using the configured cost-weighted objective and record it before final test evaluation

### Requirement: Imbalance-appropriate final evaluation
The final evaluation SHALL apply the stored threshold to the untouched test set
and report positive-class precision, recall, F1, average precision, ROC-AUC,
Brier score, confusion matrix, support, prevalence, threshold, configured costs,
and realized cost-weighted loss alongside a baseline comparison.

#### Scenario: Evaluate decision policy
- **WHEN** the evaluation stage runs with a compatible fitted artifact
- **THEN** reported classification metrics, the confusion matrix, and cost-weighted loss SHALL use the stored threshold rather than the estimator's default class prediction
