from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path
    bucket: str
    region: str
    data_path: str
    download_data: Path

@dataclass(frozen=True)
class DataValidationConfig:
    root_dir: Path
    unzip_file: Path
    status_file: Path
    all_schema: dict

@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    target_column: str
    preprocessor_path: Path
    label_encoder: Path
    categorical_columns: list
    numeric_columns: list
    columns_to_drop: list
    test_size: float
    random_state: int

@dataclass(frozen=True)
class ModelTrainerConfig:
    root_dir: Path
    train_preprocess: Path
    test_preprocess: Path
    model_name: str
    target_column: str
    n_iter: int     
    cv_folds: int
    scoring: str 
    n_jobs: int
    mlflow_username: str
    mlflow_password: str

@dataclass(frozen=True)
class ModelEvaluationConfig:
    root_dir: Path
    test_path: Path
    preprocess_path: Path
    model_path: Path
    metrics_path: str
    target_column: str
    cm_path: Path
    roc_path: Path
    mlflow_username: str
    mlflow_password: str
    experiment_name: str
    tracking_uri: str