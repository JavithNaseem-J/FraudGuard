import os
import json
import joblib
import pandas as pd
from pathlib import Path
from FraudGuard.utils.helpers import *
from FraudGuard import logger


class PredictionPipeline:
    def __init__(self):
        self.schema = read_yaml(Path('config_file/schema.yaml'))
        self.preprocessor_path = Path('artifacts/transform/preprocess/preprocessor.pkl')
        self.model_path = Path('artifacts/trainer/model.joblib')
        self.label_encoders_path = Path('artifacts/transform/preprocess/label_encoders.pkl')
        self.threshold_path = Path('artifacts/trainer/optimal_threshold.json')
        self.model_version_path = Path('artifacts/trainer/model_version.json')

        self.numerical_columns = self.schema['numeric_columns']
        self.categorical_columns = self.schema['categorical_columns']
        self.target_column = self.schema['target_column']['name']

        # Validate required files exist
        for path in [self.preprocessor_path, self.model_path, self.label_encoders_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f'CRITICAL: Required artifact missing at {path}. '
                    f'Please run the training pipeline first.'
                )
            
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        self.label_encoders = joblib.load(self.label_encoders_path)
        
        # Load optimal threshold (FAIL LOUDLY if missing/corrupted)
        self.optimal_threshold = self._load_optimal_threshold()
        
        # Load model version metadata
        self.model_version = self._load_model_version()
    
    def _load_optimal_threshold(self) -> float:
        """
        Load optimal threshold from training artifact.
        FAILS LOUDLY if threshold is missing or invalid (production safety).
        """
        if not self.threshold_path.exists():
            raise FileNotFoundError(
                f"CRITICAL: Threshold file missing at {self.threshold_path}. "
                f"Cannot proceed without optimized threshold. "
                f"Please run the training pipeline to generate this artifact."
            )
        
        try:
            with open(self.threshold_path, 'r') as f:
                threshold_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(
                f"CRITICAL: Corrupted threshold file at {self.threshold_path}: {e}. "
                f"Please regenerate by running the training pipeline."
            )
        
        if 'optimal_threshold' not in threshold_data:
            raise KeyError(
                f"CRITICAL: Missing 'optimal_threshold' key in {self.threshold_path}. "
                f"File may be corrupted. Please regenerate by running the training pipeline."
            )
        
        threshold = float(threshold_data['optimal_threshold'])
        
        # Validate threshold range
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(
                f"CRITICAL: Invalid threshold value: {threshold}. "
                f"Must be between 0 and 1. Please regenerate by running the training pipeline."
            )
        
        logger.info(f"✓ Loaded optimal threshold: {threshold}")
        return threshold
    
    def _load_model_version(self) -> dict:
        """
        Load model version metadata for tracking and audit trails.
        Returns default metadata if file doesn't exist (non-critical).
        """
        if not self.model_version_path.exists():
            logger.warning(
                f"Model version file not found at {self.model_version_path}. "
                f"Using default metadata."
            )
            return {
                "version": "unknown",
                "trained_at": "unknown",
                "model_name": "unknown"
            }
        
        try:
            with open(self.model_version_path, 'r') as f:
                version_data = json.load(f)
            logger.info(f"✓ Loaded model version: {version_data.get('version', 'unknown')}")
            return version_data
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"Failed to load model version: {e}. Using default metadata.")
            return {
                "version": "unknown",
                "trained_at": "unknown",
                "model_name": "unknown"
            }

    def preprocess_data(self, input_data):
        """Preprocess input data for prediction."""
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        
        data = input_data.copy()
        
        # Encode categorical features
        for column in self.categorical_columns:
            if column in data.columns and column in self.label_encoders:
                encoder = self.label_encoders[column]
                known_categories = set(encoder.classes_)
                data[column] = data[column].astype(str).apply(
                    lambda x: x if x in known_categories else encoder.classes_[0]
                )
                data[column] = encoder.transform(data[column].astype(str))

        # Convert to numeric
        for column in self.numerical_columns:
            if column in data.columns:
                data[column] = data[column].astype(float)

        try:
            if hasattr(self.preprocessor, 'feature_names_in_'):
                required_columns = list(self.preprocessor.feature_names_in_)
                data = data[required_columns]
            
            return self.preprocessor.transform(data)
        except Exception as e:
            raise RuntimeError(f'Error during preprocessing: {str(e)}')

    def predict(self, input_data):
        """
        Make fraud prediction with validated inputs.
        
        Args:
            input_data: pandas DataFrame with transaction features
            
        Returns:
            dict with fraud_status, fraud_probability, threshold_used, 
            confidence, and model_version
        """
        processed_data = self.preprocess_data(input_data)
        
        # Get prediction probabilities
        prediction_proba = self.model.predict_proba(processed_data)[0]
        
        # Extract probability for fraud class (class 1)
        fraud_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        
        # Use optimal threshold loaded from training artifact
        prediction = 1 if fraud_probability >= self.optimal_threshold else 0
        
        fraud_status = "Yes" if prediction == 1 else "No"
        
        # Calculate confidence based on distance from threshold
        distance_from_threshold = abs(fraud_probability - self.optimal_threshold)
        if distance_from_threshold > 0.2:
            confidence = "High"
        elif distance_from_threshold > 0.1:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        # Log prediction details
        logger.info(
            f"Prediction: {fraud_status} | "
            f"Probability: {fraud_probability:.4f} | "
            f"Threshold: {self.optimal_threshold} | "
            f"Confidence: {confidence} | "
            f"Model: {self.model_version.get('model_name', 'unknown')} "
            f"v{self.model_version.get('version', 'unknown')}"
        )

        return {
            "fraud_status": fraud_status,
            "fraud_probability": float(fraud_probability),
            "threshold_used": float(self.optimal_threshold),
            "confidence": confidence,
            "model_version": self.model_version.get("version", "unknown")
        }

