import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from FraudGuard import logger
from FraudGuard.entity.config_entity import DataTransformationConfig
from FraudGuard.utils.helpers import create_directories


class Transform:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.columns_to_drop = config.columns_to_drop
        self.target_column = config.target_column
        self.test_size = config.test_size
        self.random_state = config.random_state

    def preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate the target and remove exact duplicate transactions."""
        data = data.copy()

        if self.target_column not in data.columns:
            raise ValueError(f"Missing target column: {self.target_column}")

        target_values = set(data[self.target_column].dropna().unique())
        if not target_values or not target_values.issubset({0, 1}):
            raise ValueError(
                f"Target {self.target_column} must contain only binary 0/1 values"
            )

        rows_before = len(data)
        data = data.drop_duplicates().reset_index(drop=True)
        logger.info("Removed %s exact duplicate rows", rows_before - len(data))

        return data

    def _require_validated_data(self) -> None:
        status_path = self.config.validation_status_path
        if not status_path.exists():
            raise FileNotFoundError(
                f"Validation status not found at {status_path}; run validation first"
            )

        with status_path.open(encoding="utf-8") as status_file:
            status = json.load(status_file)

        if status.get("validation_status") is not True:
            raise ValueError("Data validation failed; refusing to create model splits")

    def train_test_splitting(self):
        """Create duplicate-free, seeded, stratified raw train/test splits."""
        self._require_validated_data()
        logger.info(f"Loading data from {self.config.data_path}")
        data = pd.read_csv(self.config.data_path)
        data = self.preprocess_data(data)

        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=y,
        )

        X_train = X_train.drop(columns=self.columns_to_drop, errors="ignore")
        X_test = X_test.drop(columns=self.columns_to_drop, errors="ignore")
        train = X_train.assign(**{self.target_column: y_train.to_numpy()})
        test = X_test.assign(**{self.target_column: y_test.to_numpy()})

        split_dir = os.path.join(self.config.root_dir, "split")
        create_directories([split_dir])
        train.to_csv(os.path.join(split_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(split_dir, "test.csv"), index=False)

        logger.info(f"Training data shape: {train.shape}")
        logger.info(f"Test data shape: {test.shape}")
        logger.info(
            "Fraud prevalence - train: %.4f, test: %.4f",
            train[self.target_column].mean(),
            test[self.target_column].mean(),
        )

        return train, test


if __name__ == "__main__":
    from FraudGuard.config.config import ConfigurationManager

    logger.info(">>>>>> Stage: Preprocess started <<<<<<")
    config = ConfigurationManager()
    transform = Transform(config=config.get_data_transformation_config())
    transform.train_test_splitting()
    logger.info(">>>>>> Stage: Preprocess completed <<<<<<")
