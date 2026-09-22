from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    kind: str
    training_path: Path
    test_path: Path | None = None
    target_path: Path | None = None
    target_column: str | None = None
    join_key: str | None = None
    license_note: str = ""
    leakage_risk_columns: tuple[str, ...] = field(default_factory=tuple)
    auxiliary_paths: tuple[Path, ...] = field(default_factory=tuple)

    def missing_paths(self) -> list[Path]:
        paths = [self.training_path, self.test_path, self.target_path]
        return [path for path in paths if path is not None and not path.exists()]


def default_dataset_registry(
    project_root: Path | None = None,
) -> dict[str, DatasetSpec]:
    root = project_root or Path(__file__).resolve().parents[3]
    return {
        "transaction-data-local": DatasetSpec(
            name="transaction-data-local",
            kind="labeled_train_unlabeled_test",
            training_path=root / "data" / "train_transaction.csv",
            test_path=None,
            target_path=None,
            target_column="isFraud",
            join_key="TransactionID",
            license_note=(
                "User-downloaded transaction benchmark files. Public test labels "
                "are unavailable, so benchmark metrics must use a split from the "
                "labeled training file. Raw data stays local."
            ),
            auxiliary_paths=(
                root / "data" / "train_identity.csv",
                root / "data" / "test_identity.csv",
            ),
        ),
    }


def validate_dataset_paths(spec: DatasetSpec) -> dict:
    missing = [str(path) for path in spec.missing_paths()]
    return {
        "name": spec.name,
        "kind": spec.kind,
        "ready": not missing,
        "missing_paths": missing,
        "target_column": spec.target_column,
        "join_key": spec.join_key,
        "auxiliary_paths": [str(path) for path in spec.auxiliary_paths],
    }


def find_registered_dataset(registry: dict[str, DatasetSpec], name: str) -> DatasetSpec:
    if name == "baseline-current":
        raise ValueError(
            "The legacy baseline dataset is retired from executable workflows; "
            "use transaction-data-local or historical documentation."
        )
    try:
        return registry[name]
    except KeyError as error:
        available = ", ".join(sorted(registry))
        raise ValueError(f"Unknown dataset '{name}'. Available: {available}") from error


def leakage_columns_present(columns: Iterable[str], spec: DatasetSpec) -> list[str]:
    column_set = set(columns)
    return sorted(
        column for column in spec.leakage_risk_columns if column in column_set
    )


def validate_dataset_contract(spec: DatasetSpec) -> dict:
    path_status = validate_dataset_paths(spec)
    if not path_status["ready"]:
        return path_status

    train = pd.read_csv(spec.training_path, nrows=5000)
    issues: list[str] = []
    if spec.target_column and spec.target_column not in train.columns:
        issues.append(f"missing training target column: {spec.target_column}")
    if spec.join_key and spec.join_key not in train.columns:
        issues.append(f"missing training join key: {spec.join_key}")
    if train.duplicated().any():
        issues.append("duplicate rows detected in training sample")

    leakage = leakage_columns_present(train.columns, spec)
    if leakage:
        issues.append(f"leakage-risk columns present: {leakage}")

    return {
        **path_status,
        "ready": not issues,
        "issues": issues,
        "sample_rows_checked": int(len(train)),
    }
