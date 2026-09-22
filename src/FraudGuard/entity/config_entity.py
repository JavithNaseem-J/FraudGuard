from pathlib import Path
from typing import Any, Dict, List
from pydantic import BaseModel, ConfigDict


class DataIngestionConfig(BaseModel):
    """Configuration for data ingestion stage."""

    model_config = ConfigDict(frozen=True)

    root_dir: Path
    local_data_path: Path
    download_data: Path


class DataValidationConfig(BaseModel):
    """Configuration for data validation stage."""

    model_config = ConfigDict(frozen=True)

    root_dir: Path
    unzip_file: Path
    status_file: Path
    all_schema: Dict[str, Any]


class DataTransformationConfig(BaseModel):
    """Configuration for data transformation stage."""

    model_config = ConfigDict(frozen=True)

    root_dir: Path
    data_path: Path
    validation_status_path: Path
    target_column: str
    columns_to_drop: List[str]
    test_size: float = 0.2
    random_state: int = 42


class ModelTrainerConfig(BaseModel):
    """Configuration for model training stage."""

    model_config = ConfigDict(frozen=True)

    root_dir: Path
    train_path: Path
    model_name: str
    target_column: str
    categorical_columns: List[str]
    numeric_columns: List[str]
    random_state: int = 42
    n_iter: int = 10
    cv_folds: int = 5
    scoring: str = "f1"
    n_jobs: int = -1
    mlflow_username: str = ""
    mlflow_password: str = ""
    false_positive_cost: float = 1.0
    false_negative_cost: float = 20.0


class ModelEvaluationConfig(BaseModel):
    """Configuration for model evaluation stage."""

    model_config = ConfigDict(frozen=True)

    root_dir: Path
    test_path: Path
    model_path: Path
    threshold_path: Path
    model_version_path: Path
    metrics_path: Path
    target_column: str
    cm_path: Path
    roc_path: Path
    pr_path: Path
    mlflow_username: str = ""
    mlflow_password: str = ""
    experiment_name: str = "Fraud-Detection"
    tracking_uri: str = ""
    false_positive_cost: float = 1.0
    false_negative_cost: float = 20.0
