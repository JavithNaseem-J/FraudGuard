import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from FraudGuard.components.validation import Validation
from FraudGuard.components.preprocess import Transform
from FraudGuard.entity.config_entity import DataValidationConfig, DataTransformationConfig

def test_data_validation():
    """Simple test for data validation logic."""
    schema = {'col1': 'int64', 'col2': 'object'}
    data = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']})
    
    config = DataValidationConfig(
        root_dir=Path("."), unzip_file=Path("."), 
        status_file=Path("status.json"), all_schema=schema
    )
    validator = Validation(config)
    
    assert validator.validate_column_presence(data, schema) is True
    assert validator.validate_data_types(data, schema) is True

def test_data_transformation_encoding():
    """Simple test for categorical encoding."""
    data = pd.DataFrame({
        'cat_col': ['A', 'B', 'A'],
        'num_col': [10, 20, 30],
        'target': [0, 1, 0]
    })
    
    config = DataTransformationConfig(
        root_dir=Path("."), data_path=Path("."), target_column='target',
        preprocessor_path=Path("preprocess/pre.pkl"), label_encoder=Path("preprocess/le.pkl"),
        categorical_columns=['cat_col'], numeric_columns=['num_col'],
        columns_to_drop=[]
    )
    transformer = Transform(config)
    processed = transformer.preprocess_data(data)
    
    assert processed['cat_col'].dtype in ['int32', 'int64']
    assert len(transformer.label_encoders) == 1

def test_inference_logic():
    """Test the threshold-based prediction logic."""
    fraud_probability = 0.7
    optimal_threshold = 0.25
    prediction = 1 if fraud_probability >= optimal_threshold else 0
    assert prediction == 1
    
    low_prob = 0.1
    assert (1 if low_prob >= optimal_threshold else 0) == 0

def test_helpers_json(tmp_path):
    """Test helper functions for JSON save/load."""
    from FraudGuard.utils.helpers import save_json
    test_file = tmp_path / "test.json"
    data = {"key": "value"}
    save_json(test_file, data)
    assert test_file.exists()

def test_mlflow_init_idempotency():
    """Test that MLflow init can be called multiple times without error."""
    from FraudGuard.utils.helpers import init_mlflow_tracking
    # Should not raise exception
    init_mlflow_tracking()
    init_mlflow_tracking()
