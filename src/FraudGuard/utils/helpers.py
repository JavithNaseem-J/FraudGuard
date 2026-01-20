import os
import json
import joblib
import boto3
import yaml
import mlflow
import dagshub
from typing import Any
from pathlib import Path
from FraudGuard.utils.logging import logger
from botocore.exceptions import ClientError
from ensure import ensure_annotations


@ensure_annotations
def download_from_s3(bucket: str, s3_path: str, local_path: Path, aws_region: str = None) -> bool:
    """Download a file from an S3 bucket."""
    try:
        profile = os.getenv('AWS_PROFILE')
        if profile:
            session = boto3.Session(profile_name=profile)
        else:
            session = boto3.Session()

        s3_client = session.client(
            's3',
            region_name=aws_region or os.getenv('AWS_REGION', 'us-east-1')
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        s3_client.download_file(bucket, s3_path, str(local_path))
        logger.info(f"Downloaded s3://{bucket}/{s3_path} to {local_path}")
        return True
    except ClientError as e:
        logger.error(f"Failed to download s3://{bucket}/{s3_path}: {str(e)}")
        return False

@ensure_annotations
def read_yaml(path_to_yaml: Path) -> dict:
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"yaml file: {path_to_yaml} loaded successfully")
            return content
    except Exception as e:
        raise e


@ensure_annotations
def create_directories(path_to_directories: list, verbose=True):
    """create list of directories

    Args:
        path_to_directories (list): list of path of directories
        ignore_log (bool, optional): ignore if multiple dirs is to be created. Defaults to False.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"created directory at: {path}")


@ensure_annotations
def save_json(path: Path, data: dict):
    """save json data

    Args:
        path (Path): path to json file
        data (dict): data to be saved in json file
    """
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logger.info(f"json file saved at: {path}")


@ensure_annotations
def load_json(path: Path) -> dict:
    with open(path) as f:
        content = json.load(f)

    logger.info(f"json file loaded succesfully from: {path}")
    return content



@ensure_annotations
def save_bin(data: object, path: Path):

    joblib.dump(value=data, filename=path)
    logger.info(f"binary file saved at: {path}")


@ensure_annotations
def load_bin(path: Path) -> Any:
    """load binary data

    Args:
        path (Path): path to binary file

    Returns:
        Any: object stored in the file
    """
    data = joblib.load(path)
    logger.info(f"binary file loaded from: {path}")
    return data




@ensure_annotations
def get_size(path: Path) -> str:
    """get size in KB

    Args:
        path (Path): path of the file

    Returns:
        str: size in KB
    """
    size_in_kb = round(os.path.getsize(path)/1024)
    return f"~ {size_in_kb} KB"


_mlflow_initialized = False

@ensure_annotations
def init_mlflow_tracking(mlflow_username: str = None, mlflow_password: str = None):
    """
    Initialize MLflow and DagsHub tracking once.
    """
    global _mlflow_initialized
    if _mlflow_initialized:
        return
    
    if mlflow_username:
        os.environ["MLFLOW_TRACKING_USERNAME"] = mlflow_username
    if mlflow_password:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = mlflow_password
    
    dagshub.init(repo_owner='JavithNaseem-J', repo_name='FraudGuard')
    mlflow.set_tracking_uri('https://dagshub.com/JavithNaseem-J/FraudGuard.mlflow')
    mlflow.set_experiment("Fraud-Detection")
    
    _mlflow_initialized = True
    logger.info("MLflow and DagsHub tracking initialized.")
