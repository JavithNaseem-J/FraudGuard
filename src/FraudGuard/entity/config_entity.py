from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel


class DataIngestionConfig(BaseModel):
    """Configuration for data ingestion stage."""
    root_dir: Path
    bucket: str
    region: str = "us-east-1"
    data_path: str
    download_data: Path

    class Config:
        frozen = True


class DataValidationConfig(BaseModel):
    """Configuration for data validation stage."""
    root_dir: Path
    unzip_file: Path
    status_file: Path
    all_schema: Dict[str, Any]

    class Config:
        frozen = True


class DataTransformationConfig(BaseModel):
    """Configuration for data transformation stage."""
    root_dir: Path
    data_path: Path
    target_column: str
    preprocessor_path: Path
    label_encoder: Path
    categorical_columns: List[str]
    numeric_columns: List[str]
    columns_to_drop: List[str]
    test_size: float = 0.2
    random_state: int = 42

    class Config:
        frozen = True


class ModelTrainerConfig(BaseModel):
    """Configuration for model training stage."""
    root_dir: Path
    train_preprocess: Path
    test_preprocess: Path
    model_name: str
    target_column: str
    n_iter: int = 10
    cv_folds: int = 5
    scoring: str = "f1"
    n_jobs: int = -1
    mlflow_username: str = ""
    mlflow_password: str = ""

    class Config:
        frozen = True


class ModelEvaluationConfig(BaseModel):
    """Configuration for model evaluation stage."""
    root_dir: Path
    test_path: Path
    preprocess_path: Path
    model_path: Path
    metrics_path: str
    target_column: str
    cm_path: Path
    roc_path: Path
    mlflow_username: str = ""
    mlflow_password: str = ""
    experiment_name: str = "Fraud-Detection"
    tracking_uri: str = ""

    class Config:
        frozen = True