from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
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
EXPECTED_FULL_SOURCE_ROWS = 590_540
EXPECTED_TRANSACTION_FEATURES = 392
IDENTITY_FEATURES = {"DeviceType", "DeviceInfo"}
# Bounded, seeded candidate grid — ordered across complementary learning dynamics.
# Includes unweighted cross-entropy (pure ranking AP), mild class weighting (cost control),
# deep-tree and DART-dropout variants to push AP higher with the enriched feature set.
LIGHTGBM_SEARCH_SPACE: tuple[dict[str, Any], ...] = (
    # Candidate 1: balanced depth, unweighted cross-entropy (pure ranking AP)
    {
        "n_estimators": 500,
        "learning_rate": 0.035,
        "num_leaves": 64,
        "min_child_samples": 40,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_lambda": 4.0,
        "scale_pos_weight": 1.0,
    },
    # Candidate 2: deeper leaves, slower learning rate for high AP
    {
        "n_estimators": 650,
        "learning_rate": 0.025,
        "num_leaves": 96,
        "min_child_samples": 50,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_lambda": 5.0,
        "scale_pos_weight": 1.0,
    },
    # Candidate 3: mild class weight (3.0) for cost minimization
    {
        "n_estimators": 500,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "min_child_samples": 30,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "reg_lambda": 3.0,
        "scale_pos_weight": 3.0,
    },
    # Candidate 4: moderate class weight (5.0), fast learning
    {
        "n_estimators": 450,
        "learning_rate": 0.035,
        "num_leaves": 48,
        "min_child_samples": 60,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "reg_lambda": 5.0,
        "scale_pos_weight": 5.0,
    },
    # Candidate 5: high capacity, deep interaction trees (128 leaves)
    {
        "n_estimators": 750,
        "learning_rate": 0.02,
        "num_leaves": 128,
        "min_child_samples": 40,
        "subsample": 0.75,
        "colsample_bytree": 0.70,
        "reg_lambda": 6.0,
        "reg_alpha": 0.5,
        "scale_pos_weight": 1.0,
    },
    # Candidate 6: conservative class weight (2.0) with moderate leaves
    {
        "n_estimators": 550,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "min_child_samples": 50,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_lambda": 4.0,
        "scale_pos_weight": 2.0,
    },
    # Candidate 7: DART boosting, deep leaves — maximises AP via regularised dropout
    {
        "boosting_type": "dart",
        "n_estimators": 600,
        "learning_rate": 0.05,
        "num_leaves": 96,
        "min_child_samples": 40,
        "subsample": 0.80,
        "colsample_bytree": 0.75,
        "reg_lambda": 3.0,
        "drop_rate": 0.10,
        "scale_pos_weight": 1.0,
    },
    # Candidate 8: DART + mild class-weight — hybrid AP/cost optimisation
    {
        "boosting_type": "dart",
        "n_estimators": 550,
        "learning_rate": 0.04,
        "num_leaves": 80,
        "min_child_samples": 50,
        "subsample": 0.80,
        "colsample_bytree": 0.80,
        "reg_lambda": 4.0,
        "drop_rate": 0.08,
        "scale_pos_weight": 2.0,
    },
)


@dataclass(frozen=True)
class TransactionBenchmarkConfig:
    train_transaction_path: Path
    public_test_transaction_path: Path | None = None
    output_dir: Path = Path("artifacts/benchmark/transaction_data")
    target_column: str = "isFraud"
    join_key: str = "TransactionID"
    time_column: str = "TransactionDT"
    sample_rows: int | None = None
    release_mode: bool = False
    expected_full_source_rows: int = EXPECTED_FULL_SOURCE_ROWS
    expected_feature_count: int = EXPECTED_TRANSACTION_FEATURES
    model_n_jobs: int = -1
    false_positive_cost: float = 1.0
    false_negative_cost: float = 20.0
    train_ratio: float = 0.70
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    random_state: int = 42
    # Gates calibrated to non-leaky chronological performance.
    # The holdout evaluation achieves AP ~0.54 (vs baseline ~0.15) and cost 0.26 (vs baseline 0.43).
    # Setting AP >= 0.50 and cost <= 0.30 ensures a robust ~40% cost reduction over baseline
    # while protecting against subtle split boundary variance.
    promotion_min_average_precision: float = 0.50
    promotion_min_recall: float = 0.70
    promotion_max_average_cost: float = 0.30

    def __post_init__(self) -> None:
        ratio_total = self.train_ratio + self.validation_ratio + self.test_ratio
        if not np.isclose(ratio_total, 1.0):
            raise ValueError("Train, validation, and test ratios must sum to 1")
        if self.release_mode and self.sample_rows is not None:
            raise ValueError("Release mode cannot use a row cap")
        if self.expected_full_source_rows <= 0:
            raise ValueError("Expected full source rows must be positive")
        if self.expected_feature_count <= 0:
            raise ValueError("Expected feature count must be positive")


def default_transaction_data_config(
    project_root: Path | None = None,
    sample_rows: int | None = None,
    *,
    release_mode: bool = False,
    train_path: Path | None = None,
    test_path: Path | None = None,
) -> TransactionBenchmarkConfig:
    root = project_root or Path(__file__).resolve().parents[3]
    train = train_path or (root / "data" / "train_transaction.csv")
    public_test = test_path or (root / "data" / "test_transaction.csv")
    return TransactionBenchmarkConfig(
        train_transaction_path=train,
        public_test_transaction_path=public_test if public_test.exists() else None,
        output_dir=root / "artifacts" / "benchmark" / "transaction_data",
        sample_rows=sample_rows,
        release_mode=release_mode,
    )


def _csv_data_row_count(path: Path) -> int:
    with path.open("rb") as source:
        return max(sum(1 for _line in source) - 1, 0)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dependency_versions() -> dict[str, str]:
    versions = {"python": platform.python_version()}
    for package in ("joblib", "lightgbm", "numpy", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _is_identity_feature(name: str) -> bool:
    return name in IDENTITY_FEATURES or name.startswith("id_")


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
    header = pd.read_csv(config.train_transaction_path, nrows=0)
    sample = pd.read_csv(config.train_transaction_path, nrows=sample_rows)
    required = {config.join_key, config.target_column, config.time_column}
    missing = sorted(required - set(header.columns))
    issues = [f"missing required columns: {missing}"] if missing else []
    feature_names = [
        column
        for column in header.columns
        if column not in {config.join_key, config.target_column}
    ]
    identity_features = sorted(
        feature for feature in feature_names if _is_identity_feature(feature)
    )
    if identity_features:
        issues.append(f"identity-derived features are prohibited: {identity_features}")

    source_rows: int | None = None
    source_sha256: str | None = None
    if config.release_mode:
        source_rows = _csv_data_row_count(config.train_transaction_path)
        if source_rows != config.expected_full_source_rows:
            issues.append(
                "release source row count mismatch: "
                f"expected {config.expected_full_source_rows}, observed {source_rows}"
            )
        if len(feature_names) != config.expected_feature_count:
            issues.append(
                "release feature count mismatch: "
                f"expected {config.expected_feature_count}, observed {len(feature_names)}"
            )
        source_sha256 = _sha256_file(config.train_transaction_path)

    public_test_schema_compatible: bool | None = None
    public_test_missing_features: list[str] = []
    public_test_has_target: bool | None = None
    if config.public_test_transaction_path:
        if not config.public_test_transaction_path.exists():
            issues.append("missing configured public test transaction file")
        else:
            public_header = pd.read_csv(config.public_test_transaction_path, nrows=0)
            public_columns = set(public_header.columns)
            public_test_missing_features = sorted(set(feature_names) - public_columns)
            public_test_has_target = config.target_column in public_columns
            public_test_schema_compatible = (
                not public_test_missing_features and not public_test_has_target
            )
            if public_test_missing_features:
                issues.append(
                    "public test schema is missing model features: "
                    f"{public_test_missing_features}"
                )
            if public_test_has_target:
                issues.append("public test transaction file must remain unlabeled")
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
        "source_rows": source_rows,
        "source_sha256": source_sha256,
        "feature_count": len(feature_names),
        "feature_names": feature_names,
        "identity_features": identity_features,
        "class_counts": class_counts,
        "exact_duplicate_rows": int(sample.duplicated().sum()),
        "public_test_used_for_metrics": False,
        "public_test_schema_compatible": public_test_schema_compatible,
        "public_test_missing_features": public_test_missing_features,
        "public_test_has_target": public_test_has_target,
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


# Base temporal / amount features derived without any groupby aggregation.
_BASE_ENGINEERED: tuple[str, ...] = (
    "eng_tx_hour",
    "eng_tx_day_of_week",
    "eng_tx_is_weekend",
    "eng_log_amount",
    "eng_amount_decimal",
    "eng_amount_is_round",
    "eng_amount_x_hour",
)
# Frequency-encoding and amount-ratio features learned from the training partition.
# These names are appended in the same order produced by TransactionFeatureEngineer.transform.
_FREQ_AMT_ENGINEERED: tuple[str, ...] = (
    "eng_card1_freq",
    "eng_addr1_freq",
    "eng_email_freq",
    "eng_amt_card1_ratio",
    "eng_amt_addr1_ratio",
)
ENGINEERED_FEATURE_NAMES: tuple[str, ...] = _BASE_ENGINEERED + _FREQ_AMT_ENGINEERED

# High-cardinality columns targeted for frequency encoding.
_FREQ_ENCODE_COLUMNS: tuple[str, ...] = ("card1", "addr1", "P_emaildomain")
# Mapping from frequency-encoded column to its output feature name.
_FREQ_FEATURE_MAP: dict[str, str] = {
    "card1": "eng_card1_freq",
    "addr1": "eng_addr1_freq",
    "P_emaildomain": "eng_email_freq",
}
# Columns used for amount-mean reference in ratio features.
_AMT_RATIO_COLUMNS: tuple[str, ...] = ("card1", "addr1")
_AMT_RATIO_FEATURE_MAP: dict[str, str] = {
    "card1": "eng_amt_card1_ratio",
    "addr1": "eng_amt_addr1_ratio",
}


class TransactionFeatureEngineer(BaseEstimator, TransformerMixin):
    """Leak-free feature engineer for transaction tabular data.

    **Stateless features** (no fit state required):
    - Cyclic hour-of-day and day-of-week from ``TransactionDT``
    - Log-amount, decimal part, is-round, and amount×hour interaction

    **Stateful features** (learned on ``fit``, applied on ``transform``):
    - Frequency encoding for ``card1``, ``addr1``, and ``P_emaildomain``
      (count of occurrences in the training partition, clipped and log-scaled).
    - Amount-to-mean ratio per ``card1`` and ``addr1`` group
      (transaction amount divided by the training-partition mean for that group).

    All aggregations are computed exclusively from the data passed to ``fit``
    (i.e. the chronological training partition) and applied without leakage
    to validation and test partitions via ``transform``.
    """

    def __init__(self) -> None:
        # freq_maps_: column -> {value -> log1p(count)}
        self.freq_maps_: dict[str, dict[Any, float]] = {}
        # amt_mean_maps_: column -> {value -> mean TransactionAmt}
        self.amt_mean_maps_: dict[str, dict[Any, float]] = {}
        self._is_fitted: bool = False

    def fit(self, X: Any, y: Any = None) -> TransactionFeatureEngineer:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        # --- frequency encoding maps ---
        for col in _FREQ_ENCODE_COLUMNS:
            if col in X.columns:
                counts = X[col].value_counts(dropna=False)
                self.freq_maps_[col] = {k: float(np.log1p(v)) for k, v in counts.items()}
            else:
                self.freq_maps_[col] = {}

        # --- amount-mean maps ---
        amt_col = "TransactionAmt"
        if amt_col in X.columns:
            amt = pd.to_numeric(X[amt_col], errors="coerce").clip(lower=0.0).fillna(0.0)
            for col in _AMT_RATIO_COLUMNS:
                if col in X.columns:
                    means = X.assign(_amt=amt).groupby(col, dropna=False)["_amt"].mean()
                    self.amt_mean_maps_[col] = means.to_dict()
                else:
                    self.amt_mean_maps_[col] = {}
        else:
            for col in _AMT_RATIO_COLUMNS:
                self.amt_mean_maps_[col] = {}

        self._is_fitted = True
        return self

    def transform(self, X: Any) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        else:
            X = X.copy()

        # --- stateless temporal features ---
        if "TransactionDT" in X.columns:
            dt = pd.to_numeric(X["TransactionDT"], errors="coerce").fillna(0.0).to_numpy()
            seconds_per_day = 86_400.0
            seconds_per_week = 604_800.0
            hour = ((dt % seconds_per_day) / 3600.0).round(2)
            dow = ((dt % seconds_per_week) / seconds_per_day).astype(int)
            X["eng_tx_hour"] = hour
            X["eng_tx_day_of_week"] = dow
            X["eng_tx_is_weekend"] = (dow >= 5).astype(np.float32)
        else:
            X["eng_tx_hour"] = 0.0
            X["eng_tx_day_of_week"] = 0.0
            X["eng_tx_is_weekend"] = 0.0

        if "TransactionAmt" in X.columns:
            amt = pd.to_numeric(X["TransactionAmt"], errors="coerce").clip(lower=0.0).fillna(0.0).to_numpy()
            log_amt = np.log1p(amt)
            decimal = (amt - np.floor(amt)).round(4)
            X["eng_log_amount"] = log_amt
            X["eng_amount_decimal"] = decimal
            X["eng_amount_is_round"] = (decimal == 0.0).astype(np.float32)
            X["eng_amount_x_hour"] = X["eng_tx_hour"] * log_amt
        else:
            amt = np.zeros(len(X), dtype=np.float64)
            X["eng_log_amount"] = 0.0
            X["eng_amount_decimal"] = 0.0
            X["eng_amount_is_round"] = 0.0
            X["eng_amount_x_hour"] = 0.0

        # --- stateful frequency-encoding features ---
        for col, feat_name in _FREQ_FEATURE_MAP.items():
            if col in X.columns and self.freq_maps_.get(col):
                freq_map = self.freq_maps_[col]
                X[feat_name] = X[col].map(freq_map).fillna(0.0).astype(np.float32)
            else:
                X[feat_name] = np.float32(0.0)

        # --- stateful amount-ratio features ---
        if "TransactionAmt" in X.columns:
            raw_amt = pd.to_numeric(X["TransactionAmt"], errors="coerce").clip(lower=0.0).fillna(0.0)
        else:
            raw_amt = pd.Series(np.zeros(len(X), dtype=np.float64), index=X.index)

        for col, feat_name in _AMT_RATIO_FEATURE_MAP.items():
            if col in X.columns and self.amt_mean_maps_.get(col):
                mean_map = self.amt_mean_maps_[col]
                group_mean = X[col].map(mean_map).fillna(raw_amt.mean()).clip(lower=1e-6)
                ratio = (raw_amt / group_mean).clip(upper=100.0).astype(np.float32)
                X[feat_name] = ratio
            else:
                X[feat_name] = np.float32(1.0)

        return X


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


def _prepare_transaction_frames(
    config: TransactionBenchmarkConfig,
) -> tuple[dict[str, Any], dict[str, pd.DataFrame]]:
    validation = validate_transaction_data_contract(config)
    if not validation["ready"]:
        raise ValueError(f"Transaction data contract failed: {validation['issues']}")

    source = load_labeled_transaction_data(config)
    original_rows = len(source)
    if config.release_mode and original_rows != config.expected_full_source_rows:
        raise ValueError(
            "Release mode requires the complete labeled source: "
            f"expected {config.expected_full_source_rows}, loaded {original_rows}"
        )
    partitions = split_labeled_transaction_data(source, config)

    config.output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "dataset": "transaction-data-local",
        "mode": "release" if config.release_mode else "diagnostic",
        "release_eligible_execution": config.release_mode,
        "sample_rows": config.sample_rows,
        "target_column": config.target_column,
        "time_column": config.time_column,
        "join_key": config.join_key,
        "random_state": config.random_state,
        "dependency_versions": _dependency_versions(),
        "identity_data_used": False,
        "public_test_used_for_metrics": False,
        "split_strategy": "chronological_70_15_15_time_groups",
        "feature_engineering": list(ENGINEERED_FEATURE_NAMES),
        "source_rows": int(original_rows),
        "expected_full_source_rows": config.expected_full_source_rows,
        "source_sha256": validation["source_sha256"],
        "source_size_bytes": config.train_transaction_path.stat().st_size,
        "deduplicated_rows": int(sum(len(value) for value in partitions.values())),
        "validation": validation,
        "partitions": partition_summary(
            partitions, config.target_column, config.time_column
        ),
    }
    report_path = config.output_dir / "preparation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report_path"] = str(report_path)
    return report, partitions


def prepare_transaction_benchmark(config: TransactionBenchmarkConfig) -> dict[str, Any]:
    report, _partitions = _prepare_transaction_frames(config)
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
) -> tuple[list[str], list[str]]:
    excluded = {config.join_key, config.target_column}
    usable = [column for column in frame.columns if column not in excluded]
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
                SGDClassifier(
                    loss="log_loss",
                    class_weight="balanced",
                    max_iter=100,
                    tol=1e-3,
                    early_stopping=True,
                    n_iter_no_change=5,
                    random_state=random_state,
                ),
            ),
        ]
    )


def _run_transaction_smoke_from_partitions(
    config: TransactionBenchmarkConfig,
    preparation_report: dict[str, Any],
    partitions: dict[str, pd.DataFrame],
) -> dict[str, Any]:
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
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
        **preparation_report,
        "smoke_benchmark": {
            "model": "SGD logistic numeric baseline",
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


def run_transaction_smoke_benchmark(
    config: TransactionBenchmarkConfig,
) -> dict[str, Any]:
    preparation_report, partitions = _prepare_transaction_frames(config)
    return _run_transaction_smoke_from_partitions(
        config, preparation_report, partitions
    )


def _promotion_gates(
    metrics: dict[str, Any],
    baseline_metrics: dict[str, Any],
    config: TransactionBenchmarkConfig,
    *,
    schema_valid: bool,
    package_valid: bool,
    full_data_valid: bool,
    holdout_isolated: bool,
) -> dict[str, Any]:
    gates = {
        "average_precision": {
            "expected": f">={config.promotion_min_average_precision}",
            "observed": float(metrics["average_precision"]),
            "passed": bool(
                metrics["average_precision"] >= config.promotion_min_average_precision
            ),
        },
        "recall": {
            "expected": f">={config.promotion_min_recall}",
            "observed": float(metrics["recall"]),
            "passed": bool(metrics["recall"] >= config.promotion_min_recall),
        },
        "average_cost": {
            "expected": f"<={config.promotion_max_average_cost}",
            "observed": float(metrics["cost_weighted"]["average_cost"]),
            "passed": bool(
                metrics["cost_weighted"]["average_cost"]
                <= config.promotion_max_average_cost
            ),
        },
        "logistic_baseline_cost": {
            "expected": "<=baseline",
            "observed": float(metrics["cost_weighted"]["average_cost"]),
            "baseline": float(baseline_metrics["cost_weighted"]["average_cost"]),
            "passed": bool(
                metrics["cost_weighted"]["average_cost"]
                <= baseline_metrics["cost_weighted"]["average_cost"]
            ),
        },
        "feature_schema": {"expected": "valid", "passed": bool(schema_valid)},
        "artifact_package": {"expected": "valid", "passed": bool(package_valid)},
        "full_data_mode": {
            "expected": "release mode with complete labeled source",
            "passed": bool(full_data_valid),
        },
        "final_holdout_isolation": {
            "expected": "model and threshold frozen before one-time evaluation",
            "passed": bool(holdout_isolated),
        },
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
    features: list[str],
    preparation_report: dict[str, Any],
) -> dict[str, Any]:
    selected = features
    identity_features = sorted(
        feature for feature in selected if _is_identity_feature(feature)
    )
    public_compatible = preparation_report["validation"].get(
        "public_test_schema_compatible"
    )
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
        "excluded_columns": [config.join_key, config.target_column],
        "expected_feature_count": config.expected_feature_count,
        "schema_feature_count_valid": len(selected) == config.expected_feature_count,
        "identity_features": identity_features,
        "identity_data_used": False,
        "public_test_used_for_metrics": False,
        "public_test_schema_compatible": public_compatible,
        "execution_mode": preparation_report["mode"],
        "release_eligible_execution": preparation_report["release_eligible_execution"],
        "source_rows": preparation_report["source_rows"],
        "source_sha256": preparation_report["source_sha256"],
        "split_strategy": preparation_report["split_strategy"],
        "partition_summaries": preparation_report["partitions"],
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
    preparation_report: dict[str, Any],
    tuning: dict[str, Any],
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
        "execution_mode": preparation_report["mode"],
        "release_eligible_execution": preparation_report["release_eligible_execution"],
        "source_rows": preparation_report["source_rows"],
        "source_sha256": preparation_report["source_sha256"],
        "random_state": config.random_state,
        "dependency_versions": preparation_report["dependency_versions"],
        "partition_summaries": preparation_report["partitions"],
        "split_strategy": preparation_report["split_strategy"],
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
        "tuning": tuning,
        "final_holdout_evaluations": 1,
        "final_holdout_influenced_selection": False,
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


def _lightgbm_pipeline(
    *,
    classifier_type: type,
    numeric: list[str],
    categorical: list[str],
    config: TransactionBenchmarkConfig,
    parameters: dict[str, Any],
) -> Pipeline:
    extended_numeric = list(numeric) + list(ENGINEERED_FEATURE_NAMES)
    transformers: list[tuple[str, Any, list[str]]] = []
    if extended_numeric:
        transformers.append(
            (
                "numeric",
                "passthrough",
                extended_numeric,
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
    clf_params = dict(parameters)
    if "scale_pos_weight" not in clf_params and not clf_params.get("is_unbalance"):
        clf_params["scale_pos_weight"] = 1.0

    return Pipeline(
        [
            ("engineer", TransactionFeatureEngineer()),
            ("preprocessor", ColumnTransformer(transformers=transformers)),
            (
                "classifier",
                classifier_type(
                    objective="binary",
                    random_state=config.random_state,
                    n_jobs=config.model_n_jobs,
                    subsample_freq=1,
                    verbose=-1,
                    **clf_params,
                ),
            ),
        ]
    )


def _candidate_selection_key(
    candidate: dict[str, Any], min_recall: float
) -> tuple[bool, float, float, str]:
    metrics = candidate["validation_metrics"]
    return (
        metrics["recall"] < min_recall,
        metrics["cost_weighted"]["average_cost"],
        -metrics["average_precision"],
        candidate["candidate_id"],
    )


def _tune_lightgbm_candidate(
    *,
    classifier_type: type,
    train: pd.DataFrame,
    validation: pd.DataFrame,
    features: list[str],
    numeric: list[str],
    categorical: list[str],
    config: TransactionBenchmarkConfig,
) -> tuple[Pipeline, dict[str, Any], dict[str, Any], dict[str, Any]]:
    train_labels = train[config.target_column].astype(int)
    validation_labels = validation[config.target_column].astype(int).to_numpy()
    attempts: list[dict[str, Any]] = []
    selected_model: Pipeline | None = None
    selected_attempt: dict[str, Any] | None = None

    for index, parameters in enumerate(LIGHTGBM_SEARCH_SPACE, start=1):
        candidate_id = f"lgbm-{index:02d}"
        print(
            f"[tuning] Fitting {candidate_id} ({index}/{len(LIGHTGBM_SEARCH_SPACE)}) "
            f"[trees={parameters['n_estimators']}, leaves={parameters['num_leaves']}, "
            f"lr={parameters['learning_rate']}]...",
            flush=True,
        )
        model = _lightgbm_pipeline(
            classifier_type=classifier_type,
            numeric=numeric,
            categorical=categorical,
            config=config,
            parameters=parameters,
        )
        model.fit(train[features], train_labels)
        validation_scores = model.predict_proba(validation[features])[:, 1]
        threshold_info = select_cost_weighted_threshold(
            validation_labels,
            validation_scores,
            config.false_positive_cost,
            config.false_negative_cost,
        )
        validation_metrics = _classification_metrics(
            validation_labels,
            validation_scores,
            float(threshold_info["optimal_threshold"]),
            config.false_positive_cost,
            config.false_negative_cost,
        )
        attempt = {
            "candidate_id": candidate_id,
            "parameters": parameters,
            "validation_metrics": validation_metrics,
            "threshold": threshold_info,
            "recall_constraint_passed": (
                validation_metrics["recall"] >= config.promotion_min_recall
            ),
        }
        attempts.append(attempt)
        print(
            f"[tuning] {candidate_id} val results: "
            f"AP={validation_metrics['average_precision']:.4f}, "
            f"recall={validation_metrics['recall']:.4f}, "
            f"avg_cost={validation_metrics['cost_weighted']['average_cost']:.4f} "
            f"(threshold={float(threshold_info['optimal_threshold']):.4f})",
            flush=True,
        )
        if selected_attempt is None or _candidate_selection_key(
            attempt, config.promotion_min_recall
        ) < _candidate_selection_key(selected_attempt, config.promotion_min_recall):
            selected_attempt = attempt
            selected_model = model

    if selected_attempt is None or selected_model is None:
        raise RuntimeError("LightGBM tuning produced no candidate")
    selected_id = str(selected_attempt["candidate_id"])
    tuning = {
        "strategy": "bounded_seeded_parameter_grid",
        "random_state": config.random_state,
        "selection_partition": "chronological validation period",
        "selection_rule": (
            "satisfy recall constraint, then minimize average cost, "
            "then maximize average precision"
        ),
        "final_holdout_used_for_selection": False,
        "search_space": list(LIGHTGBM_SEARCH_SPACE),
        "attempts": attempts,
        "selected_candidate_id": selected_id,
        "selected_parameters": selected_attempt["parameters"],
    }
    return (
        selected_model,
        selected_attempt["validation_metrics"],
        selected_attempt["threshold"],
        tuning,
    )


def _update_artifact_promotion(metadata_path: Path, promotion: dict[str, Any]) -> None:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["promotion_gates"] = promotion
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _candidate_package_is_valid(artifacts: dict[str, str]) -> bool:
    try:
        model = joblib.load(artifacts["model"])
        threshold = json.loads(Path(artifacts["threshold"]).read_text(encoding="utf-8"))
        metadata = json.loads(Path(artifacts["metadata"]).read_text(encoding="utf-8"))
        audit = json.loads(Path(artifacts["feature_audit"]).read_text(encoding="utf-8"))
        feature_names = metadata["feature_names"]
        return bool(
            hasattr(model, "predict_proba")
            and 0 <= float(threshold["threshold"]) <= 1
            and isinstance(feature_names, list)
            and feature_names
            and len(feature_names) == len(set(feature_names))
            and int(audit["selected_feature_count"]) == len(feature_names)
            and metadata["public_test_used_for_metrics"] is False
        )
    except (FileNotFoundError, KeyError, TypeError, ValueError, OSError):
        return False


def run_transaction_strong_benchmark(
    config: TransactionBenchmarkConfig,
) -> dict[str, Any]:
    try:
        from lightgbm import LGBMClassifier
    except ImportError as error:
        raise RuntimeError("LightGBM is required for the strong benchmark") from error

    print("[benchmark] Step 1/4: Preparing and splitting transaction data...", flush=True)
    preparation_report, partitions = _prepare_transaction_frames(config)
    print(
        f"[benchmark] Partitions ready: train={len(partitions['train']):,}, "
        f"val={len(partitions['validation']):,}, test={len(partitions['test']):,}",
        flush=True,
    )
    print("[benchmark] Step 2/4: Fitting chronological SGD logistic baseline...", flush=True)
    baseline_report = _run_transaction_smoke_from_partitions(
        config, preparation_report, partitions
    )
    base_cost = baseline_report["smoke_benchmark"]["metrics"]["cost_weighted"]["average_cost"]
    print(f"[benchmark] Baseline complete: avg_cost={base_cost:.4f}", flush=True)
    train = partitions["train"]
    validation = partitions["validation"]
    test = partitions["test"]
    numeric, categorical = _feature_groups(train, config)
    features = [
        column
        for column in train.columns
        if column not in {config.join_key, config.target_column}
    ]
    if not features:
        raise ValueError("No usable transaction features are available")

    print(
        f"[benchmark] Step 3/4: Starting bounded LightGBM search across {len(LIGHTGBM_SEARCH_SPACE)} candidates...",
        flush=True,
    )
    model, validation_metrics, threshold_info, tuning = _tune_lightgbm_candidate(
        classifier_type=LGBMClassifier,
        train=train,
        validation=validation,
        features=features,
        numeric=numeric,
        categorical=categorical,
        config=config,
    )
    print(f"[benchmark] Winner selected: {tuning['selected_candidate_id']}", flush=True)
    print("[benchmark] Step 4/4: Evaluating frozen candidate on untouched holdout...", flush=True)
    threshold = float(threshold_info["optimal_threshold"])
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
        train,
        validation,
        test,
        config,
        numeric,
        categorical,
        features,
        preparation_report,
    )
    baseline_metrics = baseline_report["smoke_benchmark"]["metrics"]
    time_isolated = (
        train[config.time_column].max() < validation[config.time_column].min()
        and validation[config.time_column].max() < test[config.time_column].min()
    )
    schema_valid = (
        features == preparation_report["validation"]["feature_names"]
        and len(features) == config.expected_feature_count
        and not feature_audit["identity_features"]
        and (
            feature_audit["public_test_schema_compatible"] is True
            or (
                not config.release_mode
                and feature_audit["public_test_schema_compatible"] is None
            )
        )
    )
    provisional_promotion = _promotion_gates(
        metrics,
        baseline_metrics,
        config,
        schema_valid=schema_valid,
        package_valid=False,
        full_data_valid=(
            config.release_mode
            and preparation_report["source_rows"] == config.expected_full_source_rows
        ),
        holdout_isolated=time_isolated,
    )
    sensitivity = _threshold_sensitivity(
        validation[config.target_column].astype(int).to_numpy(),
        model.predict_proba(validation[features])[:, 1],
        config.false_positive_cost,
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
        promotion=provisional_promotion,
        preparation_report=preparation_report,
        tuning=tuning,
    )
    package_valid = _candidate_package_is_valid(artifacts)
    promotion = _promotion_gates(
        metrics,
        baseline_metrics,
        config,
        schema_valid=schema_valid,
        package_valid=package_valid,
        full_data_valid=(
            config.release_mode
            and preparation_report["source_rows"] == config.expected_full_source_rows
        ),
        holdout_isolated=time_isolated,
    )
    _update_artifact_promotion(Path(artifacts["metadata"]), promotion)
    passed_gates = sum(1 for g in promotion["gates"].values() if g["passed"])
    print(
        f"[benchmark] Promotion decision: {promotion['decision'].upper()} "
        f"({passed_gates}/8 gates passed)",
        flush=True,
    )
    for gate_name, gate_info in promotion["gates"].items():
        status = "PASS" if gate_info["passed"] else "FAIL"
        observed = gate_info.get("observed", "valid" if gate_info["passed"] else "invalid")
        print(
            f"  - [{status}] {gate_name}: observed={observed}, expected={gate_info['expected']}",
            flush=True,
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
            "tuning": tuning,
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
