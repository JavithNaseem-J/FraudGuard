from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import (
    ParameterSampler,
    StratifiedKFold,
    cross_val_predict,
    cross_val_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from FraudGuard import logger
from FraudGuard.entity.config_entity import ModelTrainerConfig
from FraudGuard.utils.costs import select_cost_weighted_threshold
from FraudGuard.utils.helpers import init_mlflow_tracking, save_bin, save_json


def build_preprocessor(
    categorical_columns: list[str], numeric_columns: list[str]
) -> ColumnTransformer:
    """Build preprocessing that is cloned and fitted inside every CV fold."""
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ],
        remainder="drop",
    )


def select_f1_threshold(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Select the F1-maximizing threshold from training-only scores."""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return {
            "optimal_threshold": 0.5,
            "objective": "out_of_fold_f1",
            "precision": float(precision[0]),
            "recall": float(recall[0]),
            "f1": 0.0,
        }

    precision_at_threshold = precision[:-1]
    recall_at_threshold = recall[:-1]
    denominator = precision_at_threshold + recall_at_threshold
    f1 = np.divide(
        2 * precision_at_threshold * recall_at_threshold,
        denominator,
        out=np.zeros_like(denominator),
        where=denominator > 0,
    )
    best_index = int(np.nanargmax(f1))
    return {
        "optimal_threshold": float(thresholds[best_index]),
        "objective": "out_of_fold_f1",
        "precision": float(precision_at_threshold[best_index]),
        "recall": float(recall_at_threshold[best_index]),
        "f1": float(f1[best_index]),
    }


class Trainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.use_mlflow = bool(
            self.config.mlflow_username and self.config.mlflow_password
        )
        if self.use_mlflow:
            init_mlflow_tracking(
                mlflow_username=self.config.mlflow_username,
                mlflow_password=self.config.mlflow_password,
            )
            logger.info("MLflow tracking enabled")
        else:
            logger.warning("MLflow credentials not provided - tracking disabled")

    def _candidate_models(self, imbalance_ratio: float) -> dict:
        candidates = {
            "LogisticRegression": {
                "factory": LogisticRegression,
                "fixed": {
                    "class_weight": "balanced",
                    "max_iter": 1000,
                    "random_state": self.config.random_state,
                },
                "search": {"C": [0.05, 0.1, 0.5, 1.0, 2.0, 10.0]},
            }
        }

        try:
            from xgboost import XGBClassifier

            candidates["XGBoost"] = {
                "factory": XGBClassifier,
                "fixed": {
                    "eval_metric": "logloss",
                    "n_jobs": 1,
                    "random_state": self.config.random_state,
                    "scale_pos_weight": imbalance_ratio,
                    "verbosity": 0,
                },
                "search": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.03, 0.1, 0.2],
                    "subsample": [0.8, 1.0],
                },
            }
        except ImportError:
            logger.warning("XGBoost unavailable; skipping that candidate")

        try:
            from catboost import CatBoostClassifier

            candidates["CatBoost"] = {
                "factory": CatBoostClassifier,
                "fixed": {
                    "allow_writing_files": False,
                    "auto_class_weights": "Balanced",
                    "random_seed": self.config.random_state,
                    "thread_count": 1,
                    "verbose": 0,
                },
                "search": {
                    "iterations": [100, 200, 300],
                    "depth": [4, 6, 8],
                    "learning_rate": [0.03, 0.1, 0.2],
                },
            }
        except ImportError:
            logger.warning("CatBoost unavailable; skipping that candidate")

        return candidates

    def _pipeline(self, classifier) -> Pipeline:
        return Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        self.config.categorical_columns,
                        self.config.numeric_columns,
                    ),
                ),
                ("classifier", classifier),
            ]
        )

    def train(self):
        train_df = pd.read_csv(self.config.train_path)
        if self.config.target_column not in train_df.columns:
            raise ValueError(
                f"Missing target column in training data: {self.config.target_column}"
            )

        X_train = train_df.drop(columns=[self.config.target_column])
        y_train = train_df[self.config.target_column].astype(int)
        if set(y_train.unique()) != {0, 1}:
            raise ValueError("Training target must contain both binary classes")

        expected_features = (
            self.config.numeric_columns + self.config.categorical_columns
        )
        missing_features = sorted(set(expected_features) - set(X_train.columns))
        if missing_features:
            raise ValueError(f"Training data is missing features: {missing_features}")
        X_train = X_train[expected_features]

        cv = StratifiedKFold(
            n_splits=self.config.cv_folds,
            shuffle=True,
            random_state=self.config.random_state,
        )
        baseline = self._pipeline(
            DummyClassifier(strategy="prior", random_state=self.config.random_state)
        )
        baseline_scores = cross_val_score(
            baseline,
            X_train,
            y_train,
            scoring=self.config.scoring,
            cv=cv,
            n_jobs=self.config.n_jobs,
        )

        negative_count = int((y_train == 0).sum())
        positive_count = int((y_train == 1).sum())
        imbalance_ratio = negative_count / positive_count
        best_overall = {
            "model_name": None,
            "score": float("-inf"),
            "std": float("inf"),
            "params": None,
        }

        for model_name, model_info in self._candidate_models(imbalance_ratio).items():
            parameter_sets = list(
                ParameterSampler(
                    model_info["search"],
                    n_iter=min(
                        self.config.n_iter,
                        int(
                            np.prod(
                                [
                                    len(values)
                                    for values in model_info["search"].values()
                                ]
                            )
                        ),
                    ),
                    random_state=self.config.random_state,
                )
            )
            logger.info(
                "Evaluating %s across %s parameter sets",
                model_name,
                len(parameter_sets),
            )

            for params in parameter_sets:
                classifier = model_info["factory"](**model_info["fixed"], **params)
                scores = cross_val_score(
                    self._pipeline(classifier),
                    X_train,
                    y_train,
                    scoring=self.config.scoring,
                    cv=cv,
                    n_jobs=self.config.n_jobs,
                )
                mean_score = float(scores.mean())
                std_score = float(scores.std())
                if mean_score > best_overall["score"] or (
                    np.isclose(mean_score, best_overall["score"])
                    and std_score < best_overall["std"]
                ):
                    best_overall = {
                        "model_name": model_name,
                        "score": mean_score,
                        "std": std_score,
                        "params": params,
                    }

        if best_overall["model_name"] is None:
            raise RuntimeError("No trainable classifier candidates are available")

        selected = self._candidate_models(imbalance_ratio)[best_overall["model_name"]]
        classifier = selected["factory"](**selected["fixed"], **best_overall["params"])
        selected_pipeline = self._pipeline(classifier)

        oof_scores = cross_val_predict(
            selected_pipeline,
            X_train,
            y_train,
            cv=cv,
            method="predict_proba",
            n_jobs=self.config.n_jobs,
        )[:, 1]
        threshold_info = select_cost_weighted_threshold(
            y_train.to_numpy(),
            oof_scores,
            false_positive_cost=self.config.false_positive_cost,
            false_negative_cost=self.config.false_negative_cost,
        )
        selected_pipeline.fit(X_train, y_train)

        self.config.root_dir.mkdir(parents=True, exist_ok=True)
        model_path = self.config.root_dir / self.config.model_name
        threshold_path = self.config.root_dir / "optimal_threshold.json"
        version_path = self.config.root_dir / "model_version.json"
        best_model_info_path = self.config.root_dir / "best_model_info.json"

        save_bin(data=selected_pipeline, path=Path(model_path))
        save_json(path=Path(threshold_path), data=threshold_info)

        metadata = {
            "artifact_schema_version": 2,
            "version": "2.0.0",
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "model_name": best_overall["model_name"],
            "model_params": best_overall["params"],
            "cv_metric": self.config.scoring,
            "cv_score": float(best_overall["score"]),
            "cv_std": float(best_overall["std"]),
            "baseline_cv_score": float(baseline_scores.mean()),
            "baseline_cv_std": float(baseline_scores.std()),
            "optimal_threshold": threshold_info["optimal_threshold"],
            "threshold_objective": threshold_info["objective"],
            "false_positive_cost": self.config.false_positive_cost,
            "false_negative_cost": self.config.false_negative_cost,
            "threshold_training_metrics": {
                key: value
                for key, value in threshold_info.items()
                if key
                not in {
                    "optimal_threshold",
                    "objective",
                    "false_positive_cost",
                    "false_negative_cost",
                }
            },
            "feature_names": expected_features,
            "categorical_columns": self.config.categorical_columns,
            "numeric_columns": self.config.numeric_columns,
            "target_column": self.config.target_column,
            "training_rows": int(len(train_df)),
            "training_prevalence": float(y_train.mean()),
            "random_state": self.config.random_state,
            "score_is_calibrated": False,
        }
        save_json(path=Path(version_path), data=metadata)
        save_json(path=Path(best_model_info_path), data=metadata)

        logger.info("Best training candidate: %s", best_overall)
        logger.info("Training-only threshold: %s", threshold_info)
        return metadata


if __name__ == "__main__":
    from FraudGuard.config.config import ConfigurationManager

    logger.info(">>>>>> Stage: Training started <<<<<<")
    config = ConfigurationManager()
    trainer = Trainer(config=config.get_model_training_config())
    trainer.train()
    logger.info(">>>>>> Stage: Training completed <<<<<<")
