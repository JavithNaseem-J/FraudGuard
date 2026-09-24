# Transaction Data Benchmark Adapter

This adapter prepares local transaction fraud data without treating the public test set as labeled data.

## Local Files

Expected labeled source:

- `data/train_transaction.csv`

Identity side-table support is intentionally deferred for the current serving candidate:

- `data/train_identity.csv` optional future side table

Optional public scoring inputs:

- `data/test_transaction.csv` when available
- `data/test_identity.csv`

There is no public test target in this workspace. Precision, recall, AP, ROC-AUC, F1, and cost-weighted loss must come from deterministic internal splits of `train_transaction.csv`.

## Outputs

Prepared artifacts are written under:

```text
artifacts/benchmark/transaction_data/
```

That directory is already covered by the ignored `artifacts/` path.

## Smoke Mode

Use smoke mode before a full benchmark run. It reads a bounded number of labeled rows from `train_transaction.csv`, creates internal train/validation/test splits, and runs a numeric-only logistic regression sanity benchmark.

Example:

```powershell
$env:PYTHONPATH='src'
python -c "from FraudGuard.data.transaction_benchmark import default_transaction_data_config, run_transaction_smoke_benchmark; print(run_transaction_smoke_benchmark(default_transaction_data_config(sample_rows=5000))['benchmark_report_path'])"
```

Smoke metrics prove the preparation and evaluation path works. They are not a claim that the serving model should be replaced.

## Strong Benchmark Mode

Use strong benchmark mode after smoke mode. It trains the numeric smoke baseline and a stronger LightGBM tabular candidate, then compares them in one report. It still does not promote the model into serving.

```powershell
$env:PYTHONPATH='src'
python -c "from FraudGuard.data.transaction_benchmark import default_transaction_data_config, run_transaction_strong_benchmark; print(run_transaction_strong_benchmark(default_transaction_data_config(sample_rows=50000))['strong_benchmark_report_path'])"
```

The latest bounded transaction-only candidate run used 75,000 labeled rows from `train_transaction.csv` and evaluated on a 15,000-row internal test split. Results:

| Metric | Strong candidate |
| --- | ---: |
| Average precision / PR-AUC | 0.7090 |
| ROC-AUC | 0.9401 |
| Precision at threshold | 0.2265 |
| Recall at threshold | 0.8218 |
| F1 at threshold | 0.3551 |
| Brier score | 0.0227 |
| Cost-weighted average loss | 0.1716 |
| Feature count | 392 |

The candidate passed the local evidence gates for average precision, recall, and cost-weighted loss. It is now the selected transaction-only serving model contract for production-shaped deployment through `/predict/transactions`.

Candidate artifacts from the latest run are written to:

```text
artifacts/benchmark/transaction_data/candidate/
```

The package contains the fitted model pipeline, threshold metadata, run metadata, and a feature audit. These files are intentionally separate from historical baseline artifacts and must be promoted through an immutable release manifest before deployment.

The current candidate does not require `train_identity.csv`, `test_identity.csv`, `DeviceType`, `DeviceInfo`, or `id_*` fields. Those identity side-table fields remain a future model-version option if the deployment flow later needs device/browser features.

## Full Mode

Use full mode only after smoke mode passes and local memory is sufficient:

```powershell
$env:PYTHONPATH='src'
python -c "from FraudGuard.data.transaction_benchmark import default_transaction_data_config, prepare_transaction_benchmark; print(prepare_transaction_benchmark(default_transaction_data_config())['report_path'])"
```

Full model training should remain a separate decision because larger runs change operational cost, retraining time, and feature-distribution evidence.
