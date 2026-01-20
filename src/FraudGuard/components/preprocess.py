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
        """Preprocess data: drop columns, encode categoricals."""
        data = data.copy()

        # Drop unnecessary columns
        data.drop(columns=self.columns_to_drop, inplace=True, errors='ignore')

        # Encode categorical columns
        for column in self.categorical_columns:
            if column in data.columns:
                le = LabelEncoder()
                data[column] = le.fit_transform(data[column].astype(str))
                self.label_encoders[column] = le

        # Save label encoders
        create_directories([os.path.dirname(self.config.label_encoder)])
        save_bin(data=self.label_encoders, path=Path(self.config.label_encoder))

        return data

    def train_test_splitting(self):
        """Split data into train/test sets, then apply SMOTE-Tomek to train only (no leakage)."""
        logger.info(f"Loading data from {self.config.data_path}")
        data = pd.read_csv(self.config.data_path)

        # Preprocess data
        data = self.preprocess_data(data)
        data = data.dropna()

        # Split features and target
        X = data.drop(columns=[self.target_column])
        y = data[self.target_column]

        # Train-test split FIRST (no leakage)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.random_state
        )

        # Apply SMOTE-Tomek to train only
        logger.info("Applying SMOTE-Tomek resampling to training set only...")
        smote = SMOTETomek(tomek=TomekLinks(sampling_strategy='majority'), random_state=self.random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

        # Reconstruct dataframes
        train = pd.DataFrame(X_train_resampled, columns=X.columns)
        train[self.target_column] = y_train_resampled
        test = pd.DataFrame(X_test, columns=X.columns)
        test[self.target_column] = y_test

        # Save splits
        split_dir = os.path.join(self.config.root_dir, "split")
        create_directories([split_dir])

        train.to_csv(os.path.join(split_dir, "train.csv"), index=False)
        test.to_csv(os.path.join(split_dir, "test.csv"), index=False)

        logger.info(f"Training data shape: {train.shape}")
        logger.info(f"Test data shape: {test.shape}")

        return train, test

    def preprocess_features(self, train, test):
        """Apply StandardScaler to numerical features."""
        all_columns = [c for c in train.columns if c != self.target_column]
        numeric_cols = train[all_columns].select_dtypes(include=[np.number]).columns.tolist()

        logger.info(f"Scaling {len(numeric_cols)} numeric columns")

        num_pipeline = Pipeline(steps=[("scaler", StandardScaler())])

        preprocessor = ColumnTransformer(
            transformers=[("num", num_pipeline, numeric_cols)],
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

        # Save preprocessor
        save_bin(data=preprocessor, path=Path(self.config.preprocessor_path))

        # Save processed data
        process_dir = os.path.join(self.config.root_dir, "process")
        create_directories([process_dir])

        np.save(os.path.join(process_dir, "train_processed.npy"), train_combined)
        np.save(os.path.join(process_dir, "test_processed.npy"), test_combined)

        logger.info(f"Preprocessed train shape: {train_combined.shape}")
        logger.info(f"Preprocessed test shape: {test_combined.shape}")

        return train_processed, test_processed