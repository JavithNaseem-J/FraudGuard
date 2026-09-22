## MODIFIED Requirements

### Requirement: Unified model artifact
The deployed artifact SHALL couple learned preprocessing and the classifier in a single fitted pipeline and SHALL include compatible threshold, feature-schema, model-family, training-time, and version metadata. When transaction candidate serving is enabled, the candidate artifact package SHALL also include candidate metadata and feature audit files that are validated before serving.

#### Scenario: Load compatible artifact
- **WHEN** the prediction service starts with a complete compatible artifact set
- **THEN** it loads one fitted transformation/model pipeline and validates the metadata needed to apply it

#### Scenario: Reject incomplete artifacts
- **WHEN** the model, threshold, feature schema, or required metadata is missing or invalid
- **THEN** initialization fails clearly before a prediction is served

#### Scenario: Load compatible transaction candidate
- **WHEN** transaction candidate mode is enabled and the candidate package contains model, threshold, metadata, and feature audit artifacts
- **THEN** initialization validates the candidate feature schema and exposes candidate readiness metadata

### Requirement: Deterministic request validation
Inference SHALL reject missing required features, order accepted features by the training contract, safely process missing values and unseen categories, and ignore no required feature silently. For transaction candidate batches, row features SHALL be ordered according to candidate metadata before prediction.

#### Scenario: Missing feature
- **WHEN** a request omits a required training feature
- **THEN** prediction fails with a validation error naming the missing feature

#### Scenario: Unseen nominal value
- **WHEN** a request contains an otherwise valid unseen category
- **THEN** the fitted pipeline produces a prediction without replacing it with a different known category

#### Scenario: Transaction batch features reordered
- **WHEN** a transaction candidate request supplies features in any JSON object order
- **THEN** inference SHALL reorder them to the candidate training contract before scoring
