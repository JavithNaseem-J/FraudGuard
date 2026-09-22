from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from FraudGuard import logger


class PredictionPipeline:
    def __init__(self, artifact_root: Path | None = None):
        project_root = Path(__file__).resolve().parents[3]
        trainer_root = artifact_root or project_root / "artifacts" / "trainer"
        self.model_path = trainer_root / "model.joblib"
        self.threshold_path = trainer_root / "optimal_threshold.json"
        self.model_version_path = trainer_root / "model_version.json"

        required_paths = [
            self.model_path,
            self.threshold_path,
            self.model_version_path,
        ]
        missing_paths = [str(path) for path in required_paths if not path.exists()]
        if missing_paths:
            raise FileNotFoundError(
                "Required prediction artifacts are missing: " + ", ".join(missing_paths)
            )

        self.model = joblib.load(self.model_path)
        self.threshold_info = self._load_json(self.threshold_path)
        self.model_version = self._load_json(self.model_version_path)
        self._validate_artifact_contract()

        self.optimal_threshold = float(self.threshold_info["optimal_threshold"])
        self.feature_names = list(self.model_version["feature_names"])

    @staticmethod
    def _load_json(path: Path) -> dict:
        try:
            with path.open(encoding="utf-8") as artifact_file:
                return json.load(artifact_file)
        except json.JSONDecodeError as error:
            raise ValueError(f"Corrupted JSON artifact at {path}: {error}") from error

    def _validate_artifact_contract(self) -> None:
        if self.model_version.get("artifact_schema_version") != 2:
            raise ValueError("Unsupported or missing model artifact schema version")

        required_metadata = {
            "version",
            "model_name",
            "feature_names",
            "optimal_threshold",
            "score_is_calibrated",
        }
        missing_metadata = sorted(required_metadata - set(self.model_version))
        if missing_metadata:
            raise ValueError(f"Model metadata is missing fields: {missing_metadata}")

        threshold = float(self.threshold_info.get("optimal_threshold", -1))
        if not 0 <= threshold <= 1:
            raise ValueError(f"Invalid decision threshold: {threshold}")
        if not np.isclose(
            threshold, float(self.model_version.get("optimal_threshold", -2))
        ):
            raise ValueError("Threshold artifact does not match model metadata")
        if not hasattr(self.model, "predict_proba"):
            raise TypeError("Model artifact must provide predict_proba")

    def preprocess_data(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Validate and order input; the fitted model pipeline transforms it."""
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")

        missing_features = sorted(set(self.feature_names) - set(input_data.columns))
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")

        return input_data.loc[:, self.feature_names].copy()

    def predict(self, input_data: pd.DataFrame) -> dict:
        ordered_data = self.preprocess_data(input_data)
        class_labels = list(self.model.classes_)
        if 1 not in class_labels:
            raise ValueError("Model artifact has no positive fraud class")
        positive_index = class_labels.index(1)
        fraud_score = float(self.model.predict_proba(ordered_data)[0][positive_index])
        prediction = int(fraud_score >= self.optimal_threshold)

        distance_from_threshold = abs(fraud_score - self.optimal_threshold)
        if distance_from_threshold > 0.2:
            confidence = "High"
        elif distance_from_threshold > 0.1:
            confidence = "Medium"
        else:
            confidence = "Low"

        logger.info(
            "Prediction: %s | Score: %.4f | Threshold: %.4f | Model: %s v%s",
            "Yes" if prediction else "No",
            fraud_score,
            self.optimal_threshold,
            self.model_version["model_name"],
            self.model_version["version"],
        )

        return {
            "fraud_status": "Yes" if prediction else "No",
            "fraud_probability": fraud_score,
            "fraud_score": fraud_score,
            "threshold_used": self.optimal_threshold,
            "confidence": confidence,
            "model_version": self.model_version["version"],
            "score_is_calibrated": bool(self.model_version["score_is_calibrated"]),
        }
