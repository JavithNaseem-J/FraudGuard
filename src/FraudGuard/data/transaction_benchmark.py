from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from FraudGuard.utils.costs import cost_weighted_loss, select_cost_weighted_threshold


COST_SENSITIVITY_RATIOS = (10.0, 20.0, 50.0, 100.0)


@dataclass(frozen=True)
class TransactionBenchmarkConfig:
    train_transaction_path: Path
    public_test_transaction_path: Path | None = None
    output_dir: Path = Path("artifacts/benchmark/transaction_data")
    target_column: str = "isFraud"
    join_key: str = "TransactionID"
    time_column: str = "TransactionDT"
    sample_rows: int | None = None
    false_positive_cost: float = 1.0
    false_negative_cost: float = 20.0
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    promotion_min_average_precision: float = 0.70
    promotion_min_recall: float = 0.70
    promotion_max_average_cost: float = 0.20

    def __post_init__(self) -> None:
        ratio_total = self.train_ratio + self.validation_ratio + self.test_ratio
        if not np.isclose(ratio_total, 1.0):
            raise ValueError("Train, validation, and test ratios must sum to 1")


def default_transaction_data_config(
    project_root: Path | None = None,
    sample_rows: int | None = None,
) -> TransactionBenchmarkConfig:
    root = project_root or Path(__file__).resolve().parents[3]
    public_test = root / "data" / "test_transaction.csv"
    return TransactionBenchmarkConfig(
        train_transaction_path=root / "data" / "train_transaction.csv",
        public_test_transaction_path=public_test if public_test.exists() else None,
        output_dir=root / "artifacts" / "benchmark" / "transaction_data",
        sample_rows=sample_rows,
    )


def validate_transaction_data_contract(
    config: TransactionBenchmarkConfig,
    sample_rows: int = 5000,
) -> dict[str, Any]:
    if not config.train_transaction_path.exists():
        return {
            "ready": False,
            "issues": ["missing train_transaction.csv"],
            "warnings": [],
        }
    sample = pd.read_csv(config.train_transaction_path, nrows=sample_rows)
    required = {config.join_key, config.target_column, config.time_column}
    missing = sorted(required - set(sample.columns))
    issues = [f"missing required columns: {missing}"] if missing else []
    class_counts: dict[str, int] = {}
    if config.target_column in sample:
        labels = sample[config.target_column].dropna()
        if not set(labels.unique()).issubset({0, 1}):
            issues.append(f"{config.target_column} must contain only 0 and 1")
        if labels.nunique() < 2:
            issues.append(f"{config.target_column} sample must contain both classes")
        class_counts = {
            str(key): int(value) for key, value in labels.value_counts().items()
        }
    if config.time_column in sample and sample[config.time_column].isna().any():
        issues.append(f"{config.time_column} must not contain missing values")

    return {
        "ready": not issues,
        "issues": issues,
        "warnings": [],
        "sample_rows_checked": int(len(sample)),
        "class_counts": class_counts,
        "exact_duplicate_rows": int(sample.duplicated().sum()),
        "public_test_used_for_metrics": False,
        "public_test_note": (
            "public test rows are inference-only"
            if config.public_test_transaction_path
            else "no public test transaction file configured"
        ),
    }


def load_labeled_transaction_data(config: TransactionBenchmarkConfig) -> pd.DataFrame:
    return pd.read_csv(
        config.train_transaction_path,
        nrows=config.sample_rows,
    )


def _nearest_group_boundary(
    cumulative_rows: np.ndarray,
    target_rows: float,
    *,
    minimum_group_index: int,
    maximum_group_index: int,
) -> int:
    candidates = np.arange(minimum_group_index, maximum_group_index + 1)
    distances = np.abs(cumulative_rows[candidates] - target_rows)
    return int(candidates[int(np.argmin(distances))])


def split_labeled_transaction_data(
    frame: pd.DataFrame,
    config: TransactionBenchmarkConfig,
) -> dict[str, pd.DataFrame]:
    required = {config.target_column, config.time_column}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing chronological split columns: {missing}")

    prepared = frame.drop_duplicates().copy()
    prepared[config.target_column] = prepared[config.target_column].astype(int)
    if set(prepared[config.target_column].unique()) != {0, 1}:
        raise ValueError("Labeled transaction data must contain both binary classes")
    if prepared[config.time_column].isna().any():
        raise ValueError(f"{config.time_column} must not contain missing values")

    sort_columns = [config.time_column]
    if config.join_key in prepared:
        sort_columns.append(config.join_key)
    prepared = prepared.sort_values(sort_columns, kind="mergesort").reset_index(
        drop=True
    )

    group_sizes = (
        prepared.groupby(config.time_column, sort=True, dropna=False).size().to_numpy()
    )
    if len(group_sizes) < 3:
        raise ValueError("Chronological splitting requires at least three time groups")
    cumulative = np.cumsum(group_sizes)
    train_end_group = _nearest_group_boundary(
        cumulative,
        len(prepared) * config.train_ratio,
        minimum_group_index=0,
        maximum_group_index=len(group_sizes) - 3,
    )
    validation_end_group = _nearest_group_boundary(
        cumulative,
        len(prepared) * (config.train_ratio + config.validation_ratio),
        minimum_group_index=train_end_group + 1,
        maximum_group_index=len(group_sizes) - 2,
    )
    train_end = int(cumulative[train_end_group])
    validation_end = int(cumulative[validation_end_group])
    partitions = {
        "train": prepared.iloc[:train_end].reset_index(drop=True),
        "validation": prepared.iloc[train_end:validation_end].reset_index(drop=True),
        "test": prepared.iloc[validation_end:].reset_index(drop=True),
    }
    for name, partition in partitions.items():
        if partition.empty or partition[config.target_column].nunique() != 2:
            raise ValueError(
                f"Chronological {name} partition must contain both target classes; "
                "use more representative labeled data or revise the documented window"
            )
    return partitions


def partition_summary(
    partitions: dict[str, pd.DataFrame],
    target_column: str,
    time_column: str = "TransactionDT",
) -> dict[str, Any]:
    return {
        name: {
            "rows": int(len(partition)),
            "positive_rows": int(partition[target_column].sum()),
            "prevalence": float(partition[target_column].mean()),
            "time_min": float(partition[time_column].min()),
            "time_max": float(partition[time_column].max()),
        }
        for name, partition in partitions.items()
    }


def prepare_transaction_benchmark(config: TransactionBenchmarkConfig) -> dict[str, Any]:
    validation = validate_transaction_data_contract(config)
    if not validation["ready"]:
        raise ValueError(f"Transaction data contract failed: {validation['issues']}")

    source = load_labeled_transaction_data(config)
    original_rows = len(source)
    partitions = split_labeled_transaction_data(source, config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: dict[str, str] = {}
    for name, partition in partitions.items():
        path = config.output_dir / f"{name}.csv"
        partition.to_csv(path, index=False)
        output_paths[name] = str(path)

    report = {
        "dataset": "transaction-data-local",
        "mode": "sample" if config.sample_rows else "full",
        "sample_rows": config.sample_rows,
        "target_column": config.target_column,
        "time_column": config.time_column,
        "join_key": config.join_key,
        "identity_data_used": False,
        "public_test_used_for_metrics": False,
        "split_strategy": "chronological_70_15_15_time_groups",
        "source_rows": int(original_rows),
        "deduplicated_rows": int(sum(len(value) for value in partitions.values())),
        "validation": validation,
        "partitions": partition_summary(
            partitions, config.target_column, config.time_column
        ),
        "output_paths": output_paths,
    }
    report_path = config.output_dir / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _numeric_features(
    frame: pd.DataFrame, config: TransactionBenchmarkConfig
) -> list[str]:
    excluded = {config.join_key, config.target_column}
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]


def _feature_groups(
    frame: pd.DataFrame,
    config: TransactionBenchmarkConfig,
    max_missing_fraction: float = 0.98,
) -> tuple[list[str], list[str]]:
    excluded = {config.join_key, config.target_column}
    usable = [
        column
        for column in frame.columns
        if column not in excluded
        and frame[column].isna().mean() <= max_missing_fraction
    ]
    categorical = [
        column
        for column in usable
        if pd.api.types.is_object_dtype(frame[column])
        or pd.api.types.is_bool_dtype(frame[column])
        or isinstance(frame[column].dtype, pd.CategoricalDtype)
    ]
    numeric = [
        column
        for column in usable
        if column not in categorical and pd.api.types.is_numeric_dtype(frame[column])
    ]
    return numeric, categorical


def _classification_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    false_positive_cost: float,
    false_negative_cost: float,
) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    matrix = confusion_matrix(y_true, predictions, labels=[0, 1])
    return {
        "rows": int(len(y_true)),
        "threshold": float(threshold),
        "positive_support": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "brier_score": float(brier_score_loss(y_true, scores)),
        "confusion_matrix": matrix.tolist(),
        "cost_weighted": cost_weighted_loss(
            y_true,
            predictions,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        ),
    }


def _threshold_sensitivity(
    labels: np.ndarray,
    scores: np.ndarray,
    false_positive_cost: float,
) -> list[dict[str, Any]]:
    return [
        select_cost_weighted_threshold(
            labels,
            scores,
            false_positive_cost=false_positive_cost,
            false_negative_cost=ratio,
        )
        for ratio in COST_SENSITIVITY_RATIOS
    ]


def _baseline_pipeline(feature_names: list[str], random_state: int) -> Pipeline:
    return Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(
                    transformers=[
                        (
                            "numeric",
                            Pipeline(
                                steps=[
                                    ("imputer", SimpleImputer(strategy="median")),
                                    ("scaler", StandardScaler()),
                                ]
                            ),
                            feature_names,
                        )
                    ],
                    remainder="drop",
                ),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=500,
                    random_state=random_state,
                ),
            ),
        ]
    )


def run_transaction_smoke_benchmark(
    config: TransactionBenchmarkConfig,
) -> dict[str, Any]:
    report = prepare_transaction_benchmark(config)
    train = pd.read_csv(report["output_paths"]["train"])
    validation = pd.read_csv(report["output_paths"]["validation"])
    test = pd.read_csv(report["output_paths"]["test"])
    features = _numeric_features(train, config)
    if not features:
        raise ValueError("No numeric transaction features are available")

    model = _baseline_pipeline(features, config.random_state)
    model.fit(train[features], train[config.target_column].astype(int))
    validation_labels = validation[config.target_column].astype(int).to_numpy()
    validation_scores = model.predict_proba(validation[features])[:, 1]
    threshold_info = select_cost_weighted_threshold(
        validation_labels,
        validation_scores,
        false_positive_cost=config.false_positive_cost,
        false_negative_cost=config.false_negative_cost,
    )
    threshold = float(threshold_info["optimal_threshold"])
    validation_metrics = _classification_metrics(
        validation_labels,
        validation_scores,
        threshold,
        config.false_positive_cost,
        config.false_negative_cost,
    )
    test_labels = test[config.target_column].astype(int).to_numpy()
    test_scores = model.predict_proba(test[features])[:, 1]
    test_metrics = _classification_metrics(
        test_labels,
        test_scores,
        threshold,
        config.false_positive_cost,
        config.false_negative_cost,
    )
    benchmark_report = {
        **report,
        "smoke_benchmark": {
            "model": "LogisticRegression numeric baseline",
            "metrics_source": "untouched chronological test period",
            "threshold_selected_from": "chronological validation period",
            "validation_metrics": validation_metrics,
            "metrics": test_metrics,
            "threshold": threshold_info,
            "cost_sensitivity": _threshold_sensitivity(
                validation_labels, validation_scores, config.false_positive_cost
            ),
        },
    }
    path = config.output_dir / "smoke_benchmark_report.json"
    path.write_text(json.dumps(benchmark_report, indent=2), encoding="utf-8")
    benchmark_report["benchmark_report_path"] = str(path)
    return benchmark_report


def _promotion_gates(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    config: TransactionBenchmarkConfig,
    *,
    schema_valid: bool,
    package_valid: bool,
) -> dict[str, Any]:
    gates = {
        "average_precision": {
            "expected": f">={config.promotion_min_average_precision}",
            "observed": float(metrics["average_precision"]),
            "passed": metrics["average_precision"]
            >= config.promotion_min_average_precision,
        },
        "recall": {
            "expected": f">={config.promotion_min_recall}",
            "observed": float(metrics["recall"]),
            "passed": metrics["recall"] >= config.promotion_min_recall,
        },
        "average_cost": {
            "expected": f"<={config.promotion_max_average_cost}",
            "observed": float(metrics["cost_weighted"]["average_cost"]),
            "passed": metrics["cost_weighted"]["average_cost"]
            <= config.promotion_max_average_cost,
        },
        "logistic_baseline_cost": {
            "expected": "<=baseline",
            "observed": float(metrics["cost_weighted"]["average_cost"]),
            "baseline": float(baseline_metrics["cost_weighted"]["average_cost"]),
            "passed": metrics["cost_weighted"]["average_cost"]
            <= baseline_metrics["cost_weighted"]["average_cost"],
        },
        "feature_schema": {"expected": "valid", "passed": schema_valid},
        "artifact_package": {"expected": "valid", "passed": package_valid},
    }
    passed = all(bool(gate["passed"]) for gate in gates.values())
    return {
        "gates": gates,
        "all_gates_passed": passed,
        "decision": "approved" if passed else "blocked",
        "serving_promotion": passed,
    }


def _feature_audit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: TransactionBenchmarkConfig,
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, Any]:
    selected = numeric_features + categorical_features
    return {
        "row_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "selected_feature_count": len(selected),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "selected_features": selected,
        "identity_data_used": False,
        "public_test_used_for_metrics": False,
        "schema_risks": [
            "The public test file is unlabeled and is excluded from all quality metrics.",
            "The model score is not presented as a calibrated fraud probability.",
        ],
    }


def _write_model_artifacts(
    *,
    config: TransactionBenchmarkConfig,
    model: Pipeline,
    metrics: dict[str, Any],
    validation_metrics: dict[str, Any],
    threshold_info: dict[str, Any],
    sensitivity: list[dict[str, Any]],
    feature_audit: dict[str, Any],
    features: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
    promotion: dict[str, Any],
) -> dict[str, str]:
    # Evaluation output is intentionally separate from the active serving bundle.
    # A failed benchmark must never replace the last known-good local model.
    artifact_dir = config.output_dir / "evaluated-model"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    created_at = datetime.now(UTC).isoformat()
    threshold_metadata = {
        "threshold": float(threshold_info["optimal_threshold"]),
        "objective": threshold_info["objective"],
        "false_positive_cost": config.false_positive_cost,
        "false_negative_cost": config.false_negative_cost,
        "selected_from": "chronological validation period",
        "cost_sensitivity": sensitivity,
    }
    metadata = {
        "artifact_schema_version": 1,
        "created_at_utc": created_at,
        "model_version": created_at,
        "model_name": "LightGBM transaction fraud classifier",
        "dataset": "transaction-data-local",
        "feature_source": "train_transaction.csv",
        "target_column": config.target_column,
        "time_column": config.time_column,
        "identity_data_used": False,
        "metrics_source": "untouched chronological test period",
        "public_test_used_for_metrics": False,
        "score_is_calibrated": False,
        "promotion_gates": promotion,
        "validation_metrics": validation_metrics,
        "metrics": metrics,
        "threshold": threshold_metadata,
        "feature_count": len(features),
        "numeric_feature_count": len(numeric_features),
        "categorical_feature_count": len(categorical_features),
        "feature_names": features,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
    }
    paths = {
        "model": artifact_dir / "model.joblib",
        "threshold": artifact_dir / "threshold.json",
        "metadata": artifact_dir / "metadata.json",
        "feature_audit": artifact_dir / "feature_audit.json",
    }
    joblib.dump(model, paths["model"])
    paths["threshold"].write_text(
        json.dumps(threshold_metadata, indent=2), encoding="utf-8"
    )
    paths["metadata"].write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    paths["feature_audit"].write_text(
        json.dumps(feature_audit, indent=2), encoding="utf-8"
    )
    return {
        "artifact_dir": str(artifact_dir),
        **{key: str(value) for key, value in paths.items()},
    }


def run_transaction_strong_benchmark(
    config: TransactionBenchmarkConfig,
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise RuntimeError("LightGBM is required for the strong benchmark") from error

    baseline_report = run_transaction_smoke_benchmark(config)
    train = pd.read_csv(baseline_report["output_paths"]["train"])
    validation = pd.read_csv(baseline_report["output_paths"]["validation"])
    test = pd.read_csv(baseline_report["output_paths"]["test"])
    numeric, categorical = _feature_groups(train, config)
    features = numeric + categorical
    if not features:
        raise ValueError("No usable transaction features are available")

    transformers: list[tuple[str, Pipeline, list[str]]] = []
    if numeric:
        transformers.append(
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                numeric,
            )
        )
    if categorical:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant", fill_value="__missing__"
                            ),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(
                                handle_unknown="infrequent_if_exist",
                                min_frequency=20,
                            ),
                        ),
                    ]
                ),
                categorical,
            )
        )
    negative = int((train[config.target_column] == 0).sum())
    positive = int((train[config.target_column] == 1).sum())
    model = Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                LGBMClassifier(
                    objective="binary",
                    n_estimators=350,
                    learning_rate=0.04,
                    num_leaves=48,
                    min_child_samples=40,
                    subsample=0.85,
                    colsample_bytree=0.85,
                    reg_lambda=2.0,
                    scale_pos_weight=negative / max(positive, 1),
                    random_state=config.random_state,
                    n_jobs=1,
                    verbose=-1,
                ),
            ),
        ]
    )
    model.fit(train[features], train[config.target_column].astype(int))
    validation_labels = validation[config.target_column].astype(int).to_numpy()
    validation_scores = model.predict_proba(validation[features])[:, 1]
    threshold_info = select_cost_weighted_threshold(
        validation_labels,
        validation_scores,
        config.false_positive_cost,
        config.false_negative_cost,
    )
    threshold = float(threshold_info["optimal_threshold"])
    validation_metrics = _classification_metrics(
        validation_labels,
        validation_scores,
        threshold,
        config.false_positive_cost,
        config.false_negative_cost,
    )
    test_labels = test[config.target_column].astype(int).to_numpy()
    test_scores = model.predict_proba(test[features])[:, 1]
    metrics = _classification_metrics(
        test_labels,
        test_scores,
        threshold,
        config.false_positive_cost,
        config.false_negative_cost,
    )
    feature_audit = _feature_audit(
        train, validation, test, config, numeric, categorical
    )
    baseline_metrics = baseline_report["smoke_benchmark"]["metrics"]
    promotion = _promotion_gates(
        metrics,
        baseline_metrics,
        config,
        schema_valid=len(features) == feature_audit["selected_feature_count"],
        package_valid=True,
    )
    sensitivity = _threshold_sensitivity(
        validation_labels, validation_scores, config.false_positive_cost
    )
    artifacts = _write_model_artifacts(
        config=config,
        model=model,
        metrics=metrics,
        validation_metrics=validation_metrics,
        threshold_info=threshold_info,
        sensitivity=sensitivity,
        feature_audit=feature_audit,
        features=features,
        numeric_features=numeric,
        categorical_features=categorical,
        promotion=promotion,
    )
    report = {
        **baseline_report,
        "strong_benchmark": {
            "model": "LightGBM transaction fraud classifier",
            "metrics_source": "untouched chronological test period",
            "threshold_selected_from": "chronological validation period",
            "validation_metrics": validation_metrics,
            "metrics": metrics,
            "cost_sensitivity": sensitivity,
            "feature_audit": feature_audit,
            "promotion_gates": promotion,
            "artifacts": artifacts,
        },
        "model_comparison": {
            "baseline_model": baseline_report["smoke_benchmark"]["model"],
            "strong_model": "LightGBM transaction fraud classifier",
            "validation_average_precision_delta": validation_metrics[
                "average_precision"
            ]
            - baseline_report["smoke_benchmark"]["validation_metrics"][
                "average_precision"
            ],
            "test_average_cost_delta": metrics["cost_weighted"]["average_cost"]
            - baseline_metrics["cost_weighted"]["average_cost"],
            "promotion_decision": promotion["decision"],
        },
        "model_artifacts": artifacts,
    }
    report_path = config.output_dir / "strong_benchmark_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["strong_benchmark_report_path"] = str(report_path)
    return report
