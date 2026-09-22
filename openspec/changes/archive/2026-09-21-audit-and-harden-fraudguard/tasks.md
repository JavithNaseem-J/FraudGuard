## 1. Regression Contracts

- [x] 1.1 Add tests for duplicate removal, deterministic stratified splitting, and final-test isolation.
- [x] 1.2 Add tests for fold-safe preprocessing, missing values, and unseen categorical values.
- [x] 1.3 Add tests for training-only threshold selection and threshold-aware evaluation metrics.
- [x] 1.4 Add inference artifact, required-feature, unknown-category, threshold, and response-contract tests.

## 2. Data and Model Lifecycle

- [x] 2.1 Refactor preprocessing to validate the target, remove exact duplicates, split before learned transformations, preserve class balance, and stop producing resampled/encoded arrays.
- [x] 2.2 Build unified candidate pipelines with train-fold-local imputation, scaling, one-hot encoding, imbalance handling, and a baseline comparison.
- [x] 2.3 Make tuning deterministic and training-only, select the operating threshold from out-of-fold training scores, fit the final pipeline, and save complete model metadata.
- [x] 2.4 Refactor evaluation to apply the stored threshold on untouched test data and emit imbalance-appropriate ranking, decision, calibration, support, and baseline metrics.

## 3. Serving and Pipeline Integration

- [x] 3.1 Refactor inference to load the unified pipeline, enforce ordered required features, accept unseen categories safely, and expose consistent score/threshold/version/calibration semantics.
- [x] 3.2 Update FastAPI health and prediction behavior to use the new artifact contract while preserving compatible response fields.
- [x] 3.3 Align configuration, entity models, DVC dependencies/outputs, and generated-artifact paths with the corrected lifecycle.
- [x] 3.4 Repair CI and Docker build/runtime path mismatches that prevent the verified service from being built or health-checked.

## 4. Verification and Evidence

- [x] 4.1 Run syntax/import checks and the complete feasible test suite; fix regressions caused by this change.
- [x] 4.2 Run feasible DVC/pipeline/inference/API smoke checks and record full retraining/evaluation separately if dependencies or runtime prevent execution.
- [x] 4.3 Add evidence-based project documentation covering architecture, data/split limits, leakage protections, metrics, inference, reproducibility, limitations, and interview defense.
- [x] 4.4 Strictly validate OpenSpec, update task completion state, archive the completed change when appropriate, and produce the required nine-section audit report.
