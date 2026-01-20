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

        self.numerical_columns = self.schema['numeric_columns']
        self.categorical_columns = self.schema['categorical_columns']
        self.target_column = self.schema['target_column']['name']

        # Validate required files exist
        for path in [self.preprocessor_path, self.model_path, self.label_encoders_path]:
            if not path.exists():
                raise Exception(f'File {path} not found')
            
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        self.label_encoders = joblib.load(self.label_encoders_path)
        
        # Load optimal threshold from training artifact (with fallback)
        self.optimal_threshold = self._load_optimal_threshold()
    
    def _load_optimal_threshold(self) -> float:
        """Load optimal threshold from training artifact, fallback to default if not found."""
        default_threshold = 0.25
        
        if self.threshold_path.exists():
            try:
                with open(self.threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                threshold = float(threshold_data.get('optimal_threshold', default_threshold))
                logger.info(f"Loaded optimal threshold from artifact: {threshold}")
                return threshold
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                logger.warning(f"Failed to load threshold from {self.threshold_path}: {e}. Using default.")
                return default_threshold
        else:
            logger.warning(f"Threshold artifact not found at {self.threshold_path}. Using default: {default_threshold}")
            return default_threshold

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
        processed_data = self.preprocess_data(input_data)
        
        # Get prediction probabilities
        prediction_proba = self.model.predict_proba(processed_data)[0]
        
        # Correct probability extraction for fraud class (class 1)
        fraud_probability = prediction_proba[1] if len(prediction_proba) > 1 else prediction_proba[0]
        
        # Use optimal threshold loaded from training artifact
        prediction = 1 if fraud_probability >= self.optimal_threshold else 0
        
        fraud_status = "Yes" if prediction == 1 else "No"
        
        # Debug information
        logger.info(f"Fraud probability: {fraud_probability:.4f}, Threshold: {self.optimal_threshold}, Prediction: {fraud_status}")

        return {
            "fraud_status": fraud_status,
            "fraud_probability": float(fraud_probability),
            "threshold_used": float(self.optimal_threshold),
            "confidence": "High" if abs(fraud_probability - self.optimal_threshold) > 0.2 else "Medium" if abs(fraud_probability - self.optimal_threshold) > 0.1 else "Low"
        }
