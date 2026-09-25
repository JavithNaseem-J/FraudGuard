from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd


class TransactionPipeline:
    """Load, validate, and serve one immutable transaction-model release."""

    def __init__(self, artifact_root: Path):
        self.artifact_root = artifact_root
        self.model_path = artifact_root / "model.joblib"
        self.threshold_path = artifact_root / "threshold.json"
        self.metadata_path = artifact_root / "metadata.json"
        self.feature_audit_path = artifact_root / "feature_audit.json"

        required_paths = (
            self.model_path,
            self.threshold_path,
            self.metadata_path,
            self.feature_audit_path,
        )
        missing = [path.name for path in required_paths if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Required transaction model artifacts are missing: {missing}"
            )

        self.model = joblib.load(self.model_path)
        self.threshold_info = self._load_json(self.threshold_path)
        self.metadata = self._load_json(self.metadata_path)
        self.feature_audit = self._load_json(self.feature_audit_path)
        self._validate_artifact_contract()

        self.threshold = float(
            self.threshold_info.get(
                "threshold", self.threshold_info.get("optimal_threshold")
            )
        )
        self.feature_names = list(self.metadata["feature_names"])
        self.model_version = str(
            self.metadata.get("model_version")
            or self.metadata.get("created_at_utc")
            or "local"
        )
        self.model_name = str(
            self.metadata.get("model_name", "Transaction fraud classifier")
        )
        self.score_is_calibrated = bool(self.metadata.get("score_is_calibrated", False))

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        try:
            with path.open(encoding="utf-8") as artifact_file:
                payload = json.load(artifact_file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Corrupted transaction artifact: {path.name}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"Transaction artifact must be a JSON object: {path.name}")
        return payload

    def _validate_artifact_contract(self) -> None:
        if self.metadata.get("artifact_schema_version") != 1:
            raise ValueError("Unsupported transaction artifact schema version")
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Transaction model must provide predict_proba")

        threshold = self.threshold_info.get(
            "threshold", self.threshold_info.get("optimal_threshold")
        )
        if threshold is None or not 0 <= float(threshold) <= 1:
            raise ValueError("Transaction model threshold must be between zero and one")

        feature_names = self.metadata.get("feature_names")
        if not isinstance(feature_names, list) or not feature_names:
            raise ValueError("Transaction metadata must include feature_names")
        if len(feature_names) != len(set(feature_names)):
            raise ValueError("Transaction feature_names must be unique")

        audit_feature_count = self.feature_audit.get("selected_feature_count")
        if audit_feature_count is not None and int(audit_feature_count) != len(
            feature_names
        ):
            raise ValueError("Transaction feature audit does not match metadata")
        if self.metadata.get("public_test_used_for_metrics") is not False:
            raise ValueError("Public unlabeled test data cannot be used for metrics")

    def validate_rows(
        self, rows: list[dict[str, Any]]
    ) -> tuple[pd.DataFrame, list[str]]:
        if not rows:
            raise ValueError("At least one transaction row is required")
        if not all(isinstance(row, dict) for row in rows):
            raise TypeError("Each transaction row must be a JSON object")

        required = set(self.feature_names)
        extras: set[str] = set()
        for index, row in enumerate(rows):
            missing = sorted(required - set(row))
            if missing:
                raise ValueError(f"Row {index} missing required features: {missing}")
            extras.update(set(row) - required)

        frame = pd.DataFrame(rows)
        return frame.loc[:, self.feature_names].copy(), sorted(extras)

    def predict_rows(self, rows: list[dict[str, Any]]) -> dict[str, Any]:
        ordered, ignored_features = self.validate_rows(rows)
        class_labels = list(self.model.classes_)
        if 1 not in class_labels:
            raise ValueError("Transaction model has no positive fraud class")
        positive_index = class_labels.index(1)

        scores = self.model.predict_proba(ordered)[:, positive_index]
        results = []
        for index, score in enumerate(scores):
            fraud_score = float(score)
            is_flagged = fraud_score >= self.threshold
            results.append(
                {
                    "row_index": index,
                    "fraud_status": "Yes" if is_flagged else "No",
                    "fraud_score": fraud_score,
                    "fraud_probability": fraud_score,
                    "threshold_used": self.threshold,
                    "score_is_calibrated": self.score_is_calibrated,
                    "validation_status": "valid",
                }
            )

        return {
            "results": results,
            "ignored_features": ignored_features,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "feature_count": len(self.feature_names),
        }

    def readiness_metadata(self) -> dict[str, Any]:
        return {
            "model_loaded": True,
            "model_version": self.model_version,
            "model_name": self.model_name,
            "threshold": self.threshold,
            "score_is_calibrated": self.score_is_calibrated,
            "feature_count": len(self.feature_names),
            "public_test_used_for_metrics": False,
        }

    def release_metadata(self) -> dict[str, Any]:
        threshold_metadata = self.metadata.get("threshold", {})
        return {
            "feature_schema_summary": {
                "feature_count": len(self.feature_names),
                "numeric_feature_count": self.metadata.get("numeric_feature_count"),
                "categorical_feature_count": self.metadata.get(
                    "categorical_feature_count"
                ),
            },
            "false_positive_cost": threshold_metadata.get("false_positive_cost"),
            "false_negative_cost": threshold_metadata.get("false_negative_cost"),
            "promotion_gates": self.metadata.get("promotion_gates"),
            "released_at": self.metadata.get("created_at_utc"),
        }
