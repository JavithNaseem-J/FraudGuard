## Why

FraudGuard has a complete-looking training and serving pipeline, but its
experimental validity, threshold procedure, training-serving parity, and
reported claims have not been verified as one coherent system. Because fraud
data is highly imbalanced and leakage-sensitive, defects in those boundaries
can make apparently strong metrics unusable in production.

## What Changes

- Establish explicit, testable requirements for leakage-safe dataset splitting,
  train-only fitting, resampling, tuning, threshold selection, and final-test
  isolation.
- Make evaluation appropriate for imbalanced fraud classification and ensure
  every reported metric is reproducible from the evaluation path.
- Make online inference use the same feature schema, preprocessing artifacts,
  model contract, and validated threshold semantics as training.
- Repair critical and high-severity implementation defects discovered by the
  audit, with regression tests for corrected behavior.
- Align DVC/configuration, documentation, and project claims with the verified
  implementation and clearly identify work that requires a full retraining run.

## Capabilities

### New Capabilities

- `fraud-experiment-integrity`: Leakage-safe splitting, preprocessing,
  resampling, tuning, thresholding, and evaluation requirements for the fraud
  model lifecycle.
- `fraud-inference-contract`: Stable, validated training-to-serving feature and
  artifact contract for online predictions.
- `fraud-evidence-and-claims`: Reproducible verification evidence and rules for
  defensible documentation, interview explanations, and resume claims.

### Modified Capabilities

None. This repository has no existing OpenSpec capability specifications.

## Impact

The change may affect preprocessing, model training, evaluation, inference,
API validation, pipeline configuration, DVC dependencies/outputs, tests, and
project documentation. Generated model or evaluation artifacts will only be
updated when a feasible verified run produces them; raw source data will not be
destructively changed. Public request/response behavior will remain compatible
unless the current behavior is technically invalid.
