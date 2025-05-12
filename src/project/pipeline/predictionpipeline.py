import os
import joblib
import pandas as pd
from pathlib import Path
from project.utils.common import *

class PredictionPipeline:
    def __init__(self):
        self.schema = read_yaml(Path('config_file/schema.yaml'))
        self.preprocessor_path = Path('artifacts/data_transformation/preprocess/preprocessor.pkl')
        self.model_path = Path('artifacts/model_trainer/model.joblib')
        self.label_encoders_path = Path('artifacts/data_transformation/preprocess/label_encoders.pkl')

        self.numerical_columns = self.schema.numeric_columns
        self.categorical_columns = self.schema.categorical_columns
        self.target_column = self.schema.target_column.name

        for path in [self.preprocessor_path, self.model_path, self.label_encoders_path]:
            if not path.exists():
                raise Exception(f'File {path} not found')
            
        self.preprocessor = joblib.load(self.preprocessor_path)
        self.model = joblib.load(self.model_path)
        self.label_encoders = joblib.load(self.label_encoders_path)

    def preprocess_data(self, input_data):
        if not isinstance(input_data, pd.DataFrame):
            raise TypeError("Input data must be a pandas DataFrame")
        
        data = input_data.copy()

        required_columns = [col for col in self.preprocessor.feature_names_in_]
        missing_columns = [col for col in required_columns if col not in input_data.columns]

        if missing_columns:
            raise ValueError(f"Input data is missing required columns {missing_columns}")
        
        # Encode categorical features using saved label encoders
        for column in self.categorical_columns:
            if column in data.columns:
                if column not in self.label_encoders:
                    raise ValueError(f"No label encoder found for column '{column}'")
                encoder = self.label_encoders[column]
                unknown_labels = set(data[column].astype(str)) - set(encoder.classes_)
                if unknown_labels:
                    raise ValueError(f"Unknown categories in column '{column}': {unknown_labels}")
                data[column] = encoder.transform(data[column].astype(str))

        # Convert numerical columns to float
        for column in self.numerical_columns:
            if column in data.columns:
                try:
                    data[column] = data[column].astype(float)
                except ValueError as e:
                    raise ValueError(f"Could not convert column '{column}' to float: {str(e)}")

        try:
            preprocess_data = self.preprocessor.transform(data)
            return preprocess_data
        
        except Exception as e:
            raise RuntimeError(f'Error During Preprocess: {str(e)}')

    def predict(self, input_data):
        processed_data = self.preprocess_data(input_data)
        prediction = int(self.model.predict(processed_data)[0])
        prediction_proba = self.model.predict_proba(processed_data)[0]
        class_index = 1 if len(prediction_proba) > 1 else 0
        probability = prediction_proba[class_index]
        fraud_status = "Yes" if prediction == 1 else "No"

        return {
            "fraud_status": fraud_status,
            "fraud_probability": float(probability)
        }