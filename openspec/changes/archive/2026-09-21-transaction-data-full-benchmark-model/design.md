## Context

The numeric-only smoke benchmark on 5,000 labeled transaction rows produced AP around 0.144, ROC-AUC around 0.626, and recall around 0.136 at the selected cost-weighted threshold. That result is useful as a baseline but too weak for production fraud detection.

LightGBM is available in the current environment, while CatBoost and XGBoost are not. LightGBM is a strong fit for tabular fraud data and can handle large datasets efficiently when the feature matrix is prepared carefully.

## Goals / Non-Goals

**Goals:**

- Improve benchmark predictive performance over the numeric logistic smoke baseline.
- Use only labeled transaction training rows for metrics.
- Support sample mode first, then full mode.
- Compare baseline and stronger model metrics in one report.
- Preserve cost-weighted threshold selection and business-cost reporting.

**Non-Goals:**

- No direct serving model promotion.
- No metrics from public unlabeled test data.
- No claim of enterprise production fraud performance from a sample benchmark.
- No destructive changes to raw datasets.

## Decisions

### Decision: LightGBM as the first strong candidate

Use `LGBMClassifier` when installed because it handles sparse/high-dimensional tabular data more effectively than the current numeric logistic smoke path. If LightGBM is unavailable, the benchmark should fail clearly rather than silently pretending the weaker model is the improved result.

### Decision: One-hot categorical strategy for bounded sample mode

For the first implementation, use a bounded feature preparation path with missing-value handling, high-missing filtering, and one-hot encoding with infrequent-category grouping. This gives a strong and reproducible baseline without building a full feature store.

### Decision: Cost and recall matter more than accuracy

Report AP and ROC-AUC for ranking quality, but select operating threshold on validation scores using the configured false-positive and false-negative costs. Accuracy remains secondary because fraud data is imbalanced.

## Risks / Trade-offs

- LightGBM sample gains may not transfer to full data -> run sample first, full run later.
- One-hot encoding can grow memory -> cap categories using minimum frequency and feature filters.
- Better benchmark schema does not match current API schema -> do not promote automatically.
- Cost assumptions are approximate -> persist them in the report and keep them configurable.

## Migration Plan

1. Add strong benchmark model runner alongside the smoke benchmark.
2. Add tests with synthetic transaction data.
3. Run a bounded sample benchmark locally.
4. Compare metrics with the smoke baseline.
5. Decide separately whether to run full data and create a serving promotion change.
