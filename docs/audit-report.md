# FraudGuard Technical Audit Report

## 1. Project Verdict

FraudGuard is a tabular binary classifier that trains from transaction records,
stores a versioned model artifact, and serves transaction-level fraud decisions
through FastAPI.

The original implementation was not experimentally valid. Its category
encoders were fit before splitting, its transformed/resampled training set was
created before cross-validation, exact duplicates crossed the saved partitions,
and its final test set selected the production threshold. Evaluation then
ignored that threshold and reported default model predictions. The original
metrics and model-quality claims were therefore not trustworthy.

The repaired workflow is technically defensible for a random, same-population
transaction split: learned transformations are fold-local, the final test is
isolated, threshold selection uses training-only out-of-fold scores, evaluation
and serving share one fitted pipeline and threshold contract, and the complete
DVC pipeline was reproduced.

The current model itself is not defensible for production fraud detection. Its
test average precision is 0.04855 versus a 0.04918 prevalence baseline, with
ROC-AUC 0.48753. The corrected workflow demonstrates no predictive lift.

## 2. Assumptions / Questions

- **Assumption:** The current evaluation target is an unseen transaction drawn
  from the same population, not an unseen user. **Why safe enough:** The API is
  transaction-level and the data lacks a documented new-customer requirement.
  The limitation is explicit because users can appear in both partitions.
- **Assumption:** A random stratified split is the only executable default.
  **Why safe enough:** The dataset has hour-of-day but no absolute event date.
  No temporal claim is made.
- **Assumption:** F1 is the temporary threshold objective. **Why safe enough:**
  The repository supplies no fraud-loss matrix, false-positive budget, or review
  capacity; the objective is stored and explicitly described as replaceable.
- **Question:** What are the false-positive cost, false-negative cost, and
  analyst review capacity? **Why it matters:** They determine the economically
  valid operating threshold.
- **Question:** Must the system generalize to future transactions from known
  users, unseen users, or both? **Why it matters:** It determines temporal,
  grouped, or multi-slice deployment evaluation.
- **Question:** Do downstream consumers require calibrated risk? **Why it
  matters:** Current scores are not calibrated and cannot represent expected
  loss or literal fraud probability.

## 3. Issues Found

### [CRITICAL] Final test set selected the production threshold

- **Issue:** Training used test labels to maximize F1 and save the threshold.
- **Why wrong:** The final test became part of model-policy selection.
- **Impact:** Test evidence was optimistic and could not estimate unseen
  performance.
- **Correct approach:** Select the threshold from training-only validation or
  out-of-fold predictions, then evaluate once on untouched test data.
- **Files affected:** `src/FraudGuard/components/training.py`,
  `src/FraudGuard/components/evaluation.py`.
- **Fix made:** Threshold selection now uses training-only out-of-fold scores;
  evaluation applies the stored threshold once to test data.
- **Verification:** Full `dvc repro` completed and metrics record
  `threshold_objective: out_of_fold_f1` and threshold `0.4512575674`.

### [CRITICAL] Preprocessing and resampling contaminated cross-validation

- **Issue:** Categories were encoded before splitting; scaling and SMOTE-Tomek
  were applied before model-selection folds.
- **Why wrong:** Fold validation rows influenced preprocessing, while synthetic
  samples built from the whole training set could cross fold boundaries.
  Interpolation over label-encoded nominal categories was also semantically
  invalid.
- **Impact:** Cross-validation scores did not measure the pipeline that would
  generalize to untouched data.
- **Correct approach:** Fit all learned transformations inside each fold and use
  model-native imbalance handling for nominal mixed data.
- **Files affected:** `src/FraudGuard/components/preprocess.py`,
  `src/FraudGuard/components/training.py`, configuration, DVC, tests.
- **Fix made:** Training now evaluates full imputation/scaling/one-hot/model
  pipelines inside stratified folds; SMOTE-Tomek and precomputed arrays were
  removed; classifiers use class weights.
- **Verification:** Regression tests exercise missing/unseen values; DVC trained
  successfully from raw split CSV files.

### [HIGH] Duplicate and non-stratified split evidence

- **Issue:** The source contained 881 exact duplicates, the split was not
  stratified, and the old saved artifacts had 242 exact overlaps after feature
  preparation.
- **Why wrong:** Repeated observations can inflate evaluation and unstable class
  proportions increase variance under 4.92% prevalence.
- **Impact:** The old test set was not an independent sample.
- **Correct approach:** Remove exact duplicates before a seeded stratified split.
- **Files affected:** `src/FraudGuard/components/preprocess.py`, tests.
- **Fix made:** Preprocessing removes exact duplicate source rows and stratifies
  on the target before dropping identifiers.
- **Verification:** New train/test sizes are 40,095/10,024, both contain zero
  duplicates, cross-partition exact overlap is zero, and prevalence is
  4.923%/4.918%.

### [HIGH] Evaluation metrics hid positive-class failure

- **Issue:** The old report emphasized 94.68% accuracy and weighted metrics even
  though ROC-AUC was 0.48995; it also evaluated `model.predict()` instead of the
  production threshold.
- **Why wrong:** At about 5% prevalence, majority-class behavior dominates
  accuracy and weighted averages.
- **Impact:** The project appeared strong while ranking was worse than random.
- **Correct approach:** Separate ranking from decision-policy metrics and compare
  average precision with prevalence.
- **Files affected:** `src/FraudGuard/components/evaluation.py`, README, tests.
- **Fix made:** Evaluation now reports positive-class precision/recall/F1,
  average precision, ROC-AUC, Brier score, support, prevalence, threshold,
  confusion matrix, and baseline AP.
- **Verification:** The full run honestly reports AP `0.04855` below baseline
  `0.04918` and confusion matrix `[[742, 8789], [48, 445]]`.

### [HIGH] Training-serving artifact and category drift

- **Issue:** Serving independently loaded label encoders, a preprocessor, a
  model, a threshold, and metadata; unseen categories were silently replaced by
  the first known category.
- **Why wrong:** Separately versioned learned artifacts can become incompatible,
  and category substitution changes input meaning.
- **Impact:** Online features could differ from training features without a
  visible failure.
- **Correct approach:** Serialize one fitted preprocessing/model pipeline,
  validate its metadata/threshold contract, and use an encoder with explicit
  unknown-category handling.
- **Files affected:** `src/FraudGuard/pipeline/inference_pipeline.py`, `app.py`,
  training, configuration, DVC, tests.
- **Fix made:** Serving loads one pipeline, validates artifact schema version 2,
  feature order, threshold agreement, model version, and calibration flag.
  `OneHotEncoder(handle_unknown="ignore")` handles unseen values.
- **Verification:** Unit tests reject missing and mismatched artifacts; API smoke
  prediction with an unseen location returns HTTP 200 using model version 2.0.0.

### [MEDIUM] Missing rows were discarded wholesale

- **Issue:** `dropna()` removed every transaction with any missing feature.
- **Why wrong:** About 2,500 values were missing in several columns, so complete
  case deletion could bias data and cannot be reproduced online for a single
  request.
- **Impact:** Training distribution and serving behavior diverged.
- **Correct approach:** Fit numeric median and categorical most-frequent
  imputers on training folds.
- **Files affected:** preprocessing and training.
- **Fix made:** Fold-local imputers are part of the unified pipeline.
- **Verification:** Missing-value regression tests and full training pass.

### [MEDIUM] Pipeline, CI, and container contracts were inconsistent

- **Issue:** The feature pipeline imported a nonexistent `Ingestion`, DVC
  declared legacy artifacts, CI checked wrong artifact directories, and Docker
  referenced missing `uv.lock` and `main.py` while using the wrong Python path.
- **Why wrong:** Reproducibility and deployment commands could not execute the
  repository as written.
- **Impact:** Automated training/build paths were broken or misleading.
- **Correct approach:** Make orchestration and dependency paths match actual
  modules and artifacts.
- **Files affected:** feature pipeline, `dvc.yaml`, `dvc.lock`, CI workflow,
  Dockerfile, configuration.
- **Fix made:** Paths and stage dependencies/outputs now match the unified
  pipeline; Docker uses `requirements.lock` and `/app/src`; CI verifies the
  actual trainer artifacts.
- **Verification:** DVC fully reproduced. Docker build validation was not run
  because the Docker Desktop Linux engine was not running.

## 4. Major Architecture Changes

- Replaced disconnected encoders/preprocessor/classifier artifacts with one
  fitted pipeline plus validated threshold/version metadata.
- Replaced pre-split encoding, complete-case deletion, and pre-CV SMOTE-Tomek
  with fold-local imputation/scaling/one-hot encoding and model class weights.
- Replaced test-set threshold tuning with training-only out-of-fold selection.
- Replaced default-prediction/weighted-metric evaluation with threshold-aware
  fraud metrics and a prevalence baseline.

## 5. Verification Results

### Passed

- Global auditor skill official validation.
- OpenSpec strict change validation.
- Python compilation for source, application, and tests.
- Black formatting check.
- Ruff static lint.
- Eight regression tests.
- Full DVC ingestion, validation, preprocessing, training, and evaluation.
- Duplicate, prevalence, and cross-partition overlap diagnostics.
- Direct inference smoke test from the regenerated artifact.
- FastAPI `/health` and `/predict` smoke tests, including an unseen category.

### Failed

- Model-quality gate: test average precision `0.04855` is below the `0.04918`
  prevalence baseline; ROC-AUC is `0.48753`.

### Not executed / requires full run

- Docker image build/check: Docker Desktop's Linux engine was not running.
- XGBoost and CatBoost candidates: those optional libraries were absent from the
  execution interpreter; the full run selected and evaluated Logistic
  Regression.
- SHAP plots for the regenerated model: SHAP was absent and Logistic Regression
  is not supported by the tree-only SHAP path.
- Temporal, unseen-user, calibration, drift, and cost-policy evaluations: the
  required data/business inputs do not exist in the repository.

## 6. Final Pipeline

Validated source CSV -> exact duplicate removal -> seeded stratified raw split
-> identifiers removed -> fold-local numeric imputation/scaling and categorical
imputation/one-hot encoding -> imbalance-aware candidate CV by average precision
with baseline -> training-only out-of-fold F1 threshold -> final unified pipeline
fit on all training rows -> untouched threshold-aware test evaluation ->
versioned artifact contract -> FastAPI feature validation and scoring.

## 7. Interview Defense

- A unified pipeline prevents training-serving skew and forces preprocessing to
  participate correctly in cross-validation.
- SMOTE was removed because interpolation over encoded nominal categories was
  invalid and pre-CV resampling leaked information across folds.
- Average precision is primary because fraud prevalence is about 5%; accuracy
  and weighted metrics can obscure positive-class failure.
- Out-of-fold thresholding preserves the final test while using training data
  efficiently. F1 is explicitly temporary until business costs are known.
- The corrected model's poor result must be defended honestly: methodology can
  be sound while the available features contain insufficient predictive signal.
- The present split estimates random transaction-level generalization, not
  future-time or unseen-user performance.

## 8. Resume Claim Check

### SAFE TO CLAIM

- Built a DVC-orchestrated fraud training/evaluation workflow with duplicate-safe
  stratified splitting and training-only model/threshold selection.
- Coupled preprocessing and classification in a versioned inference pipeline
  with unknown-category and artifact-contract validation.
- Added threshold-aware imbalanced-class evaluation and regression/API tests.

### REPHRASE

- “Real-time fraud detection” -> “FastAPI online inference prototype for a
  transaction fraud classifier.”
- “Optimized fraud model” -> “Cross-validated model-selection and threshold
  workflow; the current dataset did not demonstrate lift over baseline.”

### DO NOT CLAIM

- High model accuracy or effective fraud detection.
- Production-ready, calibrated risk scoring, temporal robustness, unseen-user
  generalization, live monitoring, or demonstrated financial-loss reduction.

## 9. Remaining Work

- Obtain dated transactions and define whether deployment targets known or
  unseen users; replace the random split with temporal and relevant grouped
  evaluation.
- Establish error costs or review capacity and replace F1 threshold selection
  with the business decision objective.
- Improve or validate features and label quality before adding model complexity.
- Add calibration only if downstream use requires probability semantics.
- Run the container build when Docker Desktop is available and run optional tree
  candidates in the declared complete dependency environment.
