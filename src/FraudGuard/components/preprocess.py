import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek
import joblib
from FraudGuard import logger
from FraudGuard.entity.config_entity import DataTransformationConfig
from FraudGuard.utils.helpers import create_directories, save_bin

class Transform:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.columns_to_drop = config.columns_to_drop
        self.target_column = config.target_column
        self.label_encoders = {}
        self.categorical_columns = config.categorical_columns
        self.numerical_columns = config.numeric_columns
        self.test_size = config.test_size
        self.random_state = config.random_state

    def preprocess_data(self, data):
            data = data.copy()

            data.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')

            for column in self.categorical_columns:
                if column in data.columns:
                    le = LabelEncoder()
                    data[column] = le.fit_transform(data[column].astype(str))
                    self.label_encoders[column] = le

            # Create directory and save label encoders using utils.common
            create_directories([os.path.dirname(self.config.label_encoder)])
            save_bin(data=self.label_encoders, path=Path(self.config.label_encoder))

            return data


    def train_test_splitting(self):

            logger.info(f"Loading data from {self.config.data_path}")
            data = pd.read_csv(self.config.data_path)

            data = self.preprocess_data(data)
            data = data.dropna()

            X = data.drop(columns=[self.target_column])
            y = data[self.target_column]

            smote = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'))
            X_resampled, y_resampled = smote.fit_resample(X, y)

            X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
            resampled_data = X_resampled.copy()
            resampled_data[self.target_column] = y_resampled

            train, test = train_test_split(resampled_data, test_size=self.test_size, random_state=self.random_state)

            split_dir = os.path.join(self.config.root_dir, "split")
            create_directories([split_dir])

            train_path = os.path.join(split_dir, "train.csv")
            test_path = os.path.join(split_dir, "test.csv")

            train.to_csv(train_path, index=False)
            test.to_csv(test_path, index=False)

            logger.info("Split data into training and test sets")
            logger.info(f"Training data shape: {train.shape}")
            logger.info(f"Test data shape: {test.shape}")

            return train, test


    def preprocess_features(self, train, test):
            
            numerical_columns = self.numerical_columns
            categorical_columns = self.categorical_columns.copy()

            if self.target_column in categorical_columns:
                categorical_columns.remove(self.target_column)

            logger.info(f"Numerical columns: {list(numerical_columns)}")
            logger.info(f"Categorical columns: {list(categorical_columns)}")

            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipeline, numerical_columns)
                ],
                remainder="passthrough"
            )

            train_x = train.drop(columns=[self.target_column])
            test_x = test.drop(columns=[self.target_column])
            train_y = train[self.target_column].values.reshape(-1, 1)
            test_y = test[self.target_column].values.reshape(-1, 1)

            train_processed = preprocessor.fit_transform(train_x)
            test_processed = preprocessor.transform(test_x)

            train_combined = np.hstack((train_processed, train_y))
            test_combined = np.hstack((test_processed, test_y))

            # Save preprocessor using save_bin
            save_bin(data=preprocessor, path=Path(self.config.preprocessor_path))

            # Create directory for processed data
            process_dir = os.path.join(self.config.root_dir, "process")
            create_directories([process_dir])

            train_processed_path = os.path.join(process_dir, "train_processed.npy")
            test_processed_path = os.path.join(process_dir, "test_processed.npy")

            np.save(train_processed_path, train_combined)
            np.save(test_processed_path, test_combined)

            logger.info("Preprocessed train and test data saved successfully.")
            return train_processed, test_processed
