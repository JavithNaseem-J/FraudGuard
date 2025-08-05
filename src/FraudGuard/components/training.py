import os
import json
import joblib
import mlflow
import pandas as pd
import numpy as np
import optuna
from pathlib import Path
import dagshub
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from FraudGuard import logger
from FraudGuard.utils.helpers import save_json, save_bin
from FraudGuard.entity.config_entity import ModelTrainerConfig


class Trainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

        os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config.mlflow_password

        dagshub.init(repo_owner="JavithNaseem-J", repo_name="FraudGuard")
        mlflow.set_tracking_uri("https://dagshub.com/JavithNaseem-J/FraudGuard.mlflow")
        mlflow.set_experiment("Fraud-Detection")

        self.models = {
            "XGBoost": {
                "class": XGBClassifier,
                "search_space": lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "verbosity": 0,
                    "use_label_encoder": False
                },
                "mlflow_module": mlflow.xgboost,
            },
            "CatBoost": {
                "class": CatBoostClassifier,
                "search_space": lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 5, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "verbose": 0,
                    "allow_writing_files": False
                },
                "mlflow_module": mlflow.catboost,
            },
            "LightGBM": {
                "class": LGBMClassifier,
                "search_space": lambda trial: {
                    "num_leaves": trial.suggest_int("num_leaves", 20, 150),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "verbosity": -1
                },
                "mlflow_module": mlflow.lightgbm,
            },
        }

    def train(self):
        train_data = np.load(self.config.train_preprocess, allow_pickle=True)
        test_data = np.load(self.config.test_preprocess, allow_pickle=True)

        train_x = train_data[:, :-1]
        train_y = train_data[:, -1]

        best_overall = {"model_name": None, "score": 0, "std": 0, "params": None}

        for model_name, model_info in self.models.items():
            logger.info(f"Starting HPO for: {model_name}")

            def objective(trial):
                params = model_info["search_space"](trial)
                model = model_info["class"](**params)
                cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
                scores = cross_val_score(
                    model, train_x, train_y,
                    scoring=self.config.scoring,
                    cv=cv,
                    n_jobs=self.config.n_jobs
                )
                return scores.mean()

            study = optuna.create_study(direction="maximize")
            study.optimize(objective, n_trials=self.config.n_iter)

            best_params = study.best_params
            best_score = study.best_value
            best_model = model_info["class"](**best_params)
            cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
            best_scores = cross_val_score(best_model, train_x, train_y, scoring=self.config.scoring, cv=cv)
            best_std = best_scores.std()

            # Log best model per algorithm
            with mlflow.start_run(run_name=f"{model_name}_best"):
                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_score", best_score)
                mlflow.log_metric("best_cv_std", best_std)
                mlflow.set_tags({
                    "model_name": model_name,
                    "stage": "HPO-Best"
                })

            if (
                best_score > best_overall["score"] or 
                (best_score == best_overall["score"] and best_std < best_overall["std"])
            ):
                best_overall.update({
                    "model_name": model_name,
                    "score": best_score,
                    "std": best_std,
                    "params": best_params
                })

        # Final best model
        best_model_class = self.models[best_overall["model_name"]]["class"]
        final_params = best_overall["params"]

        # Adjust verbosity for final model training
        if best_overall["model_name"] == "XGBoost":
            final_params["verbosity"] = 1
        elif best_overall["model_name"] == "CatBoost":
            final_params["verbose"] = 100
        elif best_overall["model_name"] == "LightGBM":
            final_params["verbosity"] = 1

        best_model = best_model_class(**final_params)
        best_model.fit(train_x, train_y)

        # Log & register the final best model
        with mlflow.start_run(run_name=f"{best_overall['model_name']}_final"):
            mlflow.log_params(final_params)
            mlflow.log_metric("best_cv_score", best_overall["score"])
            mlflow.log_metric("best_cv_std", best_overall["std"])
            mlflow.set_tags({"model_name": best_overall["model_name"], "stage": "final"})

            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                registered_model_name='FraudGuardModel',
            )

            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            save_bin(data=best_model, path=Path(model_path))

            best_model_info_path = os.path.join(self.config.root_dir, "best_model_info.json")
            save_json(path=Path(best_model_info_path), data=best_overall)
            mlflow.log_artifact(best_model_info_path)

        logger.info(f"Best model overall in the Model Training: {best_overall}")
        return best_overall
