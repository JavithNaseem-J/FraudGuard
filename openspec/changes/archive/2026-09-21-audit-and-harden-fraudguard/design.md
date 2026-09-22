## Context

FraudGuard is a tabular binary fraud classifier with DVC stages for ingestion,
validation, preprocessing, training, and evaluation, plus a FastAPI prediction
service. The dataset has 51,000 rows, a 4.92% positive rate, 881 exact duplicate
rows, repeated users, nominal categorical features, and no absolute event date.

The current implementation encodes categories before splitting, drops all rows
with missing values, applies SMOTE-Tomek to the entire training partition before
cross-validation, tunes the decision threshold on the final test set, evaluates
the model's default class prediction instead of that threshold, and serves from
separately managed encoder, preprocessor, model, and threshold artifacts. The
committed test split also contains exact overlaps with the resampled training
artifact. Existing metrics therefore do not form trustworthy evidence for the
deployed decision policy.

The repository has no event date that can support temporal backtesting and no
documented business cost matrix or review-capacity target. The defensible
default is therefore a duplicate-free stratified transaction split, with its
deployment limitation recorded explicitly.

## Goals / Non-Goals

**Goals:**

- Keep the final test partition isolated from preprocessing fit, model/tuning
  selection, and threshold selection.
- Ensure every learned transformation used during validation is fitted inside
  its fold and the exact fitted transformation is used during inference.
- Handle missing and unknown categorical values without discarding production
  rows or inventing ordinal meaning.
- Address class imbalance without synthetic categorical interpolation or
  cross-fold resampling contamination.
- Select a reproducible threshold from training-only out-of-fold predictions
  and evaluate that same decision policy once on the test set.
- Report metrics appropriate to severe imbalance and compare against a simple
  baseline.
- Couple model, preprocessing, feature schema, threshold, and version metadata
  so incompatible artifacts fail clearly.
- Add regression tests and align DVC, container, CI, and project claims with the
  verified implementation.

**Non-Goals:**

- Claim temporal generalization, unseen-customer generalization, calibrated
  financial risk, or production readiness without corresponding data/evidence.
- Invent fraud-review capacity or false-positive/false-negative costs.
- Add monitoring platforms, feature stores, online retraining, or new cloud
  services.
- Destructively modify the source dataset.

## Decisions

### 1. Use a unified fitted pipeline artifact

Training will construct a pipeline containing column selection,
`SimpleImputer`, `StandardScaler`, `OneHotEncoder(handle_unknown="ignore")`, and
the selected classifier. The entire fitted pipeline will be serialized as the
model artifact and loaded directly by evaluation and inference.

This eliminates drift among separately fitted label encoders, preprocessors,
and classifiers. One-hot encoding is preferred to `LabelEncoder` because the
categorical features are nominal and unseen categories need an explicit safe
path. Keeping the current separate artifacts was rejected because it permits
version skew and required category fallback to an unrelated first class.

### 2. Split before all learned transformations and remove exact duplicates

The preprocessing stage will validate the target, remove exact duplicate rows,
then create a seeded stratified train/test split. Identifier columns will be
available for duplicate/split diagnostics and removed before model fitting.

A temporal split is preferable for genuine future fraud performance, but this
dataset contains only hour-of-day, not an event date. A group split by `User_ID`
would estimate unseen-customer performance, which is not the service's stated
transaction-level contract. The chosen split estimates random unseen
transactions from the same population; this limitation must be documented.

### 3. Remove SMOTE-Tomek and use model-native imbalance handling

Synthetic interpolation over label-encoded nominal categories produces invalid
feature values, and the current resampling occurs before CV. XGBoost will use a
training-derived `scale_pos_weight`; CatBoost will use balanced class weights.
Model selection will use average precision, which reflects ranking quality under
imbalance. This is simpler and preserves the observed data distribution.

### 4. Keep preprocessing inside cross-validation

Each candidate estimator is a full pipeline. Hyperparameter search and model
comparison operate on raw training rows through stratified folds, ensuring
imputers, scaling, and category vocabularies are fitted only on each fold's
training subset. The final selected pipeline is then fitted once on all training
data.

### 5. Select the threshold from training-only out-of-fold predictions

After selecting model family and hyperparameters, the system will generate
out-of-fold probabilities on the training set and select the F1-maximizing
threshold. This is a generic operating point because no business cost/review
capacity was provided. The threshold and selection objective will be stored as
metadata. The final test set will not participate in this choice.

A dedicated validation split was considered but rejected because out-of-fold
predictions use limited data more efficiently while maintaining isolation from
the final test. Nested model-family selection remains approximated rather than
fully nested due project size and runtime; the final test remains unbiased.

### 6. Evaluate ranking and decision behavior separately

Evaluation will apply the stored threshold and report positive-class precision,
recall, F1, average precision/PR-AUC, ROC-AUC, Brier score, confusion matrix, test
support/prevalence, and the threshold. A stratified dummy baseline will be
reported for context. Accuracy and weighted metrics may be retained only as
secondary diagnostics, not as evidence of fraud detection quality.

### 7. Enforce an inference contract

The model artifact metadata will include ordered feature names, target name,
model family, training timestamp, CV score, threshold, and the statement that
the score is not guaranteed calibrated. Inference will validate required
columns, reject missing columns, order inputs deterministically, let the fitted
encoder handle unknown categories, and apply the stored threshold.

The existing response keys remain compatible. Documentation will call the
numeric value a model fraud score unless calibration evidence supports a literal
risk probability.

### 8. Treat generated artifacts as versioned evidence

DVC dependencies and outputs will match the files actually produced. Legacy
preprocessor/label-encoder artifacts will no longer be serving dependencies.
Training will not depend on test arrays. Tests will cover duplicate-free split
isolation, train-only pipeline behavior, threshold selection, metric semantics,
artifact validation, and unknown-category inference.

## Risks / Trade-offs

- **Random split cannot measure future drift** -> State this limitation and do
  not claim temporal validation; require dated data for a future upgrade.
- **Same users can appear across train/test** -> Define the current target as
  transaction-level generalization for an existing-user population and avoid
  unseen-customer claims.
- **F1 threshold is not a business-optimal policy** -> Store the objective and
  make the selector replaceable when costs or queue capacity are known.
- **Class weighting can distort probability calibration** -> Report Brier score,
  mark scores as not guaranteed calibrated, and avoid risk-probability claims.
- **Tree-library dependencies may be unavailable locally** -> Keep diagnostics
  importable where practical, run all feasible tests, and distinguish full
  retraining from static/unit verification.
- **Unified artifact invalidates legacy files** -> Regenerate artifacts through
  DVC and make missing/incompatible metadata fail clearly.

## Migration Plan

1. Add regression tests that express split, threshold, metric, and inference
   contracts.
2. Refactor preprocessing to clean, deduplicate, and stratify raw rows without
   fitting encoders or resampling.
3. Refactor training to build full candidate pipelines, tune on training only,
   create out-of-fold threshold evidence, fit the final model, and save metadata.
4. Refactor evaluation and inference to consume the unified artifact and the
   same threshold.
5. Update configuration, DVC, CI/container paths, and documentation.
6. Run tests and feasible pipeline stages. Only replace committed generated
   metrics/models when a complete compatible training run succeeds.

Rollback is a source-control revert plus restoration of the prior DVC artifacts.
No raw data migration is required.

## Open Questions

- What false-positive cost, false-negative cost, or analyst review capacity
  should replace F1 as the operating-policy objective?
- Is production success defined on future transactions from existing users,
  previously unseen users, or both? Dated data and this answer are required for
  the final deployment split strategy.
- Is calibrated risk required by downstream consumers? If so, calibration must
  be selected and evaluated on training-only validation evidence.
