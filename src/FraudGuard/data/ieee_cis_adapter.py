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
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from FraudGuard import logger
from FraudGuard.utils.costs import cost_weighted_loss, select_cost_weighted_threshold


@dataclass(frozen=True)
class IeeeCisBenchmarkConfig:
    train_transaction_path: Path
    train_identity_path: Path | None = None
    public_test_transaction_path: Path | None = None
    public_test_identity_path: Path | None = None
    output_dir: Path = Path("artifacts/benchmark/transaction_data")
    target_column: str = "isFraud"
    join_key: str = "TransactionID"
    random_state: int = 42
    validation_size: float = 0.2
    test_size: float = 0.2
    sample_rows: int | None = None
    false_positive_cost: float = 1.0
    false_negative_cost: float = 20.0
    join_identity: bool = True
    promotion_min_average_precision: float = 0.70
    promotion_min_recall: float = 0.70
    promotion_max_average_cost: float = 0.20


def default_ieee_cis_config(
    project_root: Path | None = None,
    sample_rows: int | None = None,
) -> IeeeCisBenchmarkConfig:
    root = project_root or Path(__file__).resolve().parents[3]
    public_test_transaction = root / "data" / "test_transaction.csv"
    return IeeeCisBenchmarkConfig(
        train_transaction_path=root / "data" / "train_transaction.csv",
        train_identity_path=root / "data" / "train_identity.csv",
        public_test_transaction_path=(
            public_test_transaction if public_test_transaction.exists() else None
        ),
        public_test_identity_path=root / "data" / "test_identity.csv",
        output_dir=root / "artifacts" / "benchmark" / "transaction_data",
        sample_rows=sample_rows,
    )


def _read_csv(
    path: Path, nrows: int | None = None, usecols: list[str] | None = None
) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows, usecols=usecols)


def validate_ieee_cis_contract(
    config: IeeeCisBenchmarkConfig,
    sample_rows: int = 5000,
) -> dict[str, Any]:
    issues: list[str] = []
    warnings: list[str] = []

    if not config.train_transaction_path.exists():
        issues.append(
            f"missing train transaction file: {config.train_transaction_path}"
        )
        return {"ready": False, "issues": issues, "warnings": warnings}

    sample = _read_csv(config.train_transaction_path, nrows=sample_rows)
    required_columns = {config.join_key, config.target_column}
    missing_required = sorted(required_columns - set(sample.columns))
    if missing_required:
        issues.append(f"missing required columns: {missing_required}")

    if config.target_column in sample.columns:
        labels = sample[config.target_column].dropna().unique()
        if not set(labels).issubset({0, 1}):
            issues.append(f"{config.target_column} must be binary 0/1")
        class_counts = sample[config.target_column].value_counts().to_dict()
        if len(class_counts) < 2:
            issues.append(f"{config.target_column} sample must contain both classes")
    else:
        class_counts = {}

    duplicate_ids = 0
    if config.join_key in sample.columns:
        duplicate_ids = int(sample[config.join_key].duplicated().sum())
        if duplicate_ids:
            issues.append(f"duplicate {config.join_key} values in training sample")

    identity_coverage = None
    if config.join_identity and config.train_identity_path:
        if not config.train_identity_path.exists():
            warnings.append(
                f"missing train identity file: {config.train_identity_path}"
            )
        else:
            identity_keys = _read_csv(
                config.train_identity_path, usecols=[config.join_key]
            )
            identity_key_set = set(identity_keys[config.join_key])
            identity_coverage = float(
                sample[config.join_key].isin(identity_key_set).mean()
            )

    if (
        config.public_test_transaction_path
        and config.public_test_transaction_path.exists()
    ):
        public_test_note = (
            "public test transaction file configured; labels are not used for metrics"
        )
    else:
        public_test_note = "no public test transaction file configured"

    return {
        "ready": not issues,
        "issues": issues,
        "warnings": warnings,
        "sample_rows_checked": int(len(sample)),
        "class_counts": {str(key): int(value) for key, value in class_counts.items()},
        "duplicate_training_ids": duplicate_ids,
        "identity_coverage_in_sample": identity_coverage,
        "public_test_used_for_metrics": False,
        "public_test_note": public_test_note,
    }


def load_labeled_transactions(config: IeeeCisBenchmarkConfig) -> pd.DataFrame:
    transactions = _read_csv(config.train_transaction_path, nrows=config.sample_rows)
    if not config.join_identity or not config.train_identity_path:
        return transactions
    if not config.train_identity_path.exists():
        logger.warning("Training identity file is missing; using transaction data only")
        return transactions

    identity = _read_csv(config.train_identity_path)
    before_rows = len(transactions)
    prepared = transactions.merge(
        identity,
        on=config.join_key,
        how="left",
        suffixes=("", "_identity"),
        validate="one_to_one",
    )
    if len(prepared) != before_rows:
        raise ValueError("Identity join changed transaction row count")
    return prepared


def split_labeled_transactions(
    frame: pd.DataFrame,
    config: IeeeCisBenchmarkConfig,
) -> dict[str, pd.DataFrame]:
    if config.target_column not in frame.columns:
        raise ValueError(f"Missing target column: {config.target_column}")
    y = frame[config.target_column].astype(int)
    if set(y.unique()) != {0, 1}:
        raise ValueError("Labeled transaction data must contain both binary classes")

    train_val, benchmark_test = train_test_split(
        frame,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )
    relative_validation_size = config.validation_size / (1.0 - config.test_size)
    train, validation = train_test_split(
        train_val,
        test_size=relative_validation_size,
        random_state=config.random_state,
        stratify=train_val[config.target_column].astype(int),
    )
    return {
        "train": train.sort_values(config.join_key).reset_index(drop=True),
        "validation": validation.sort_values(config.join_key).reset_index(drop=True),
        "test": benchmark_test.sort_values(config.join_key).reset_index(drop=True),
    }


def partition_summary(partitions: dict[str, pd.DataFrame], target_column: str) -> dict:
    return {
        name: {
            "rows": int(len(partition)),
            "positive_rows": int(partition[target_column].sum()),
            "prevalence": float(partition[target_column].mean()),
        }
        for name, partition in partitions.items()
    }


def prepare_ieee_cis_benchmark(config: IeeeCisBenchmarkConfig) -> dict[str, Any]:
    validation = validate_ieee_cis_contract(config)
    if not validation["ready"]:
        raise ValueError(f"Transaction data contract failed: {validation['issues']}")

    prepared = load_labeled_transactions(config)
    partitions = split_labeled_transactions(prepared, config)
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
        "join_key": config.join_key,
        "identity_join_enabled": bool(config.join_identity),
        "public_test_used_for_metrics": False,
        "validation": validation,
        "partitions": partition_summary(partitions, config.target_column),
        "output_paths": output_paths,
    }
    report_path = config.output_dir / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report


def _numeric_features(frame: pd.DataFrame, config: IeeeCisBenchmarkConfig) -> list[str]:
    excluded = {config.join_key, config.target_column}
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]


def _baseline_comparison(output_dir: Path) -> dict[str, Any]:
    project_root = output_dir.parents[2] if len(output_dir.parents) >= 3 else Path.cwd()
    baseline_path = project_root / "artifacts" / "evaluation" / "metrics.json"
    comparison: dict[str, Any] = {
        "baseline_metrics_path": str(baseline_path),
        "baseline_metrics_available": baseline_path.exists(),
        "not_directly_comparable": True,
        "reason": (
            "The current serving baseline and transaction benchmark use different "
            "schemas. Transaction benchmark evidence must not be treated as serving-model "
            "promotion without a separate promotion change."
        ),
    }
    if baseline_path.exists():
        try:
            baseline_metrics = json.loads(baseline_path.read_text(encoding="utf-8"))
            comparison["baseline_summary"] = {
                key: baseline_metrics.get(key)
                for key in [
                    "average_precision",
                    "roc_auc",
                    "brier_score",
                    "precision",
                    "recall",
                    "f1",
                    "prevalence",
                    "threshold",
                    "cost_weighted",
                ]
                if key in baseline_metrics
            }
        except json.JSONDecodeError as error:
            comparison["baseline_read_error"] = str(error)
    return comparison


def run_ieee_cis_smoke_benchmark(config: IeeeCisBenchmarkConfig) -> dict[str, Any]:
    report = prepare_ieee_cis_benchmark(config)
    train = pd.read_csv(report["output_paths"]["train"])
    validation = pd.read_csv(report["output_paths"]["validation"])
    test = pd.read_csv(report["output_paths"]["test"])

    feature_names = _numeric_features(train, config)
    if not feature_names:
        raise ValueError(
            "No numeric transaction benchmark features available for smoke run"
        )

    pipeline = Pipeline(
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
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    pipeline.fit(train[feature_names], train[config.target_column].astype(int))
    validation_scores = pipeline.predict_proba(validation[feature_names])[:, 1]
    threshold_info = select_cost_weighted_threshold(
        validation[config.target_column].astype(int).to_numpy(),
        validation_scores,
        false_positive_cost=config.false_positive_cost,
        false_negative_cost=config.false_negative_cost,
    )

    test_scores = pipeline.predict_proba(test[feature_names])[:, 1]
    threshold = float(threshold_info["optimal_threshold"])
    test_predictions = (test_scores >= threshold).astype(int)
    y_test = test[config.target_column].astype(int).to_numpy()
    matrix = confusion_matrix(y_test, test_predictions, labels=[0, 1])
    metrics = {
        "rows": int(len(test)),
        "feature_count": int(len(feature_names)),
        "threshold": threshold,
        "threshold_objective": threshold_info["objective"],
        "positive_support": int(y_test.sum()),
        "prevalence": float(y_test.mean()),
        "precision": float(precision_score(y_test, test_predictions, zero_division=0)),
        "recall": float(recall_score(y_test, test_predictions, zero_division=0)),
        "f1": float(f1_score(y_test, test_predictions, zero_division=0)),
        "average_precision": float(average_precision_score(y_test, test_scores)),
        "roc_auc": float(roc_auc_score(y_test, test_scores)),
        "brier_score": float(brier_score_loss(y_test, test_scores)),
        "confusion_matrix": matrix.tolist(),
        "cost_weighted": cost_weighted_loss(
            y_test,
            test_predictions,
            false_positive_cost=config.false_positive_cost,
            false_negative_cost=config.false_negative_cost,
        ),
    }

    benchmark_report = {
        **report,
        "smoke_benchmark": {
            "model": "LogisticRegression numeric-only smoke benchmark",
            "serving_promotion": False,
            "metrics_source": "internal labeled test split",
            "metrics": metrics,
        },
        "baseline_comparison": _baseline_comparison(config.output_dir),
    }
    benchmark_path = config.output_dir / "smoke_benchmark_report.json"
    benchmark_path.write_text(json.dumps(benchmark_report, indent=2), encoding="utf-8")
    benchmark_report["benchmark_report_path"] = str(benchmark_path)
    return benchmark_report


def _feature_groups(
    frame: pd.DataFrame,
    config: IeeeCisBenchmarkConfig,
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


def _promotion_gates(
    metrics: dict[str, Any], config: IeeeCisBenchmarkConfig
) -> dict[str, Any]:
    gates = {
        "average_precision": {
            "operator": ">=",
            "threshold": float(config.promotion_min_average_precision),
            "observed": float(metrics["average_precision"]),
            "passed": bool(
                metrics["average_precision"] >= config.promotion_min_average_precision
            ),
        },
        "recall": {
            "operator": ">=",
            "threshold": float(config.promotion_min_recall),
            "observed": float(metrics["recall"]),
            "passed": bool(metrics["recall"] >= config.promotion_min_recall),
        },
        "average_cost": {
            "operator": "<=",
            "threshold": float(config.promotion_max_average_cost),
            "observed": float(metrics["cost_weighted"]["average_cost"]),
            "passed": bool(
                metrics["cost_weighted"]["average_cost"]
                <= config.promotion_max_average_cost
            ),
        },
    }
    return {
        "gates": gates,
        "all_gates_passed": all(gate["passed"] for gate in gates.values()),
        "decision": "candidate_only",
        "serving_promotion": False,
        "reason": (
            "Promotion gates are benchmark evidence only. Serving promotion requires "
            "a separate compatibility change because benchmark features differ from "
            "the current API serving schema."
        ),
    }


def _feature_audit(
    train: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    config: IeeeCisBenchmarkConfig,
    numeric_features: list[str],
    categorical_features: list[str],
    max_missing_fraction: float = 0.98,
) -> dict[str, Any]:
    excluded_columns = [config.join_key, config.target_column]
    selected_features = numeric_features + categorical_features
    missing_fraction = train.drop(
        columns=[column for column in excluded_columns if column in train]
    )
    high_missing = (
        missing_fraction.isna()
        .mean()
        .loc[lambda series: series > max_missing_fraction]
        .sort_values(ascending=False)
    )
    return {
        "row_counts": {
            "train": int(len(train)),
            "validation": int(len(validation)),
            "test": int(len(test)),
        },
        "total_columns": int(len(train.columns)),
        "selected_feature_count": int(len(selected_features)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "excluded_columns": excluded_columns,
        "high_missing_columns": {
            column: float(value) for column, value in high_missing.head(50).items()
        },
        "dropped_high_missing_column_count": int(len(high_missing)),
        "selected_features": selected_features,
        "public_test_used_for_metrics": False,
        "schema_risks": [
            "Benchmark candidate uses the transaction benchmark feature schema, "
            "which is wider than the current API serving schema.",
            "Identity columns are sparse; missing-value behavior is part of the fitted pipeline.",
        ],
    }


def _write_candidate_artifacts(
    *,
    config: IeeeCisBenchmarkConfig,
    candidate: Pipeline,
    metrics: dict[str, Any],
    threshold_info: dict[str, Any],
    feature_audit: dict[str, Any],
    feature_names: list[str],
    numeric_features: list[str],
    categorical_features: list[str],
) -> dict[str, str]:
    candidate_dir = config.output_dir / "candidate"
    candidate_dir.mkdir(parents=True, exist_ok=True)

    model_path = candidate_dir / "model.joblib"
    threshold_path = candidate_dir / "threshold.json"
    metadata_path = candidate_dir / "metadata.json"
    feature_audit_path = candidate_dir / "feature_audit.json"

    gates = _promotion_gates(metrics, config)
    threshold_metadata = {
        "threshold": float(threshold_info["optimal_threshold"]),
        "objective": threshold_info["objective"],
        "false_positive_cost": float(config.false_positive_cost),
        "false_negative_cost": float(config.false_negative_cost),
        "selected_from": "validation split",
    }
    metadata = {
        "artifact_schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "model_name": "LightGBM tabular transaction benchmark",
        "dataset": "transaction-data-local",
        "mode": "sample" if config.sample_rows else "full",
        "sample_rows": config.sample_rows,
        "target_column": config.target_column,
        "join_key": config.join_key,
        "metrics_source": "internal labeled test split",
        "public_test_used_for_metrics": False,
        "serving_promotion": False,
        "promotion_decision": gates["decision"],
        "promotion_gates": gates,
        "metrics": metrics,
        "threshold": threshold_metadata,
        "feature_count": int(len(feature_names)),
        "numeric_feature_count": int(len(numeric_features)),
        "categorical_feature_count": int(len(categorical_features)),
        "feature_names": feature_names,
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "serving_compatibility_note": (
            "This package is a benchmark candidate. It is not loaded by the API "
            "until a separate serving-promotion change maps the serving schema."
        ),
    }

    joblib.dump(candidate, model_path)
    threshold_path.write_text(
        json.dumps(threshold_metadata, indent=2), encoding="utf-8"
    )
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    feature_audit_path.write_text(json.dumps(feature_audit, indent=2), encoding="utf-8")

    return {
        "candidate_dir": str(candidate_dir),
        "model": str(model_path),
        "threshold": str(threshold_path),
        "metadata": str(metadata_path),
        "feature_audit": str(feature_audit_path),
    }


def run_transaction_strong_benchmark(config: IeeeCisBenchmarkConfig) -> dict[str, Any]:
    """Run a stronger LightGBM benchmark and compare it with the smoke baseline."""
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise RuntimeError("LightGBM is required for the strong benchmark") from error

    baseline_report = run_ieee_cis_smoke_benchmark(config)
    train = pd.read_csv(baseline_report["output_paths"]["train"])
    validation = pd.read_csv(baseline_report["output_paths"]["validation"])
    test = pd.read_csv(baseline_report["output_paths"]["test"])

    numeric_features, categorical_features = _feature_groups(train, config)
    if not numeric_features and not categorical_features:
        raise ValueError("No usable transaction benchmark features available")

    transformers = []
    if numeric_features:
        transformers.append(
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                    ]
                ),
                numeric_features,
            )
        )
    if categorical_features:
        transformers.append(
            (
                "categorical",
                Pipeline(
                    steps=[
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
                                sparse_output=True,
                            ),
                        ),
                    ]
                ),
                categorical_features,
            )
        )

    negative_count = int((train[config.target_column] == 0).sum())
    positive_count = int((train[config.target_column] == 1).sum())
    scale_pos_weight = negative_count / max(positive_count, 1)
    candidate = Pipeline(
        steps=[
            (
                "preprocessor",
                ColumnTransformer(transformers=transformers, remainder="drop"),
            ),
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
                    scale_pos_weight=scale_pos_weight,
                    random_state=config.random_state,
                    n_jobs=1,
                    verbose=-1,
                ),
            ),
        ]
    )

    feature_names = numeric_features + categorical_features
    candidate.fit(train[feature_names], train[config.target_column].astype(int))
    validation_scores = candidate.predict_proba(validation[feature_names])[:, 1]
    threshold_info = select_cost_weighted_threshold(
        validation[config.target_column].astype(int).to_numpy(),
        validation_scores,
        false_positive_cost=config.false_positive_cost,
        false_negative_cost=config.false_negative_cost,
    )
    test_scores = candidate.predict_proba(test[feature_names])[:, 1]
    metrics = _classification_metrics(
        test[config.target_column].astype(int).to_numpy(),
        test_scores,
        float(threshold_info["optimal_threshold"]),
        false_positive_cost=config.false_positive_cost,
        false_negative_cost=config.false_negative_cost,
    )
    metrics["threshold_objective"] = threshold_info["objective"]
    metrics["feature_count"] = int(len(feature_names))
    metrics["numeric_feature_count"] = int(len(numeric_features))
    metrics["categorical_feature_count"] = int(len(categorical_features))
    feature_audit = _feature_audit(
        train,
        validation,
        test,
        config,
        numeric_features,
        categorical_features,
    )
    promotion = _promotion_gates(metrics, config)
    candidate_artifacts = _write_candidate_artifacts(
        config=config,
        candidate=candidate,
        metrics=metrics,
        threshold_info=threshold_info,
        feature_audit=feature_audit,
        feature_names=feature_names,
        numeric_features=numeric_features,
        categorical_features=categorical_features,
    )

    baseline_metrics = baseline_report["smoke_benchmark"]["metrics"]
    best_candidate = (
        "lightgbm_strong"
        if metrics["average_precision"] >= baseline_metrics["average_precision"]
        else "numeric_logistic_smoke"
    )
    report = {
        **baseline_report,
        "strong_benchmark": {
            "model": "LightGBM tabular transaction benchmark",
            "serving_promotion": False,
            "metrics_source": "internal labeled test split",
            "metrics": metrics,
            "candidate_artifacts": candidate_artifacts,
            "feature_audit": feature_audit,
            "promotion_gates": promotion,
        },
        "model_comparison": {
            "baseline_model": baseline_report["smoke_benchmark"]["model"],
            "strong_model": "LightGBM tabular transaction benchmark",
            "best_candidate_by_average_precision": best_candidate,
            "average_precision_delta": float(
                metrics["average_precision"] - baseline_metrics["average_precision"]
            ),
            "recall_delta": float(metrics["recall"] - baseline_metrics["recall"]),
            "f1_delta": float(metrics["f1"] - baseline_metrics["f1"]),
            "promotion_decision": "not_promoted",
            "promotion_reason": (
                "Benchmark evidence does not automatically replace serving artifacts; "
                "a separate promotion change must verify serving compatibility."
            ),
        },
        "candidate_artifacts": candidate_artifacts,
    }
    benchmark_path = config.output_dir / "strong_benchmark_report.json"
    benchmark_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["strong_benchmark_report_path"] = str(benchmark_path)
    return report
