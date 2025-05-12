# model_training.py
import os
import json
import joblib
import mlflow
import pandas as pd
from pathlib import Path
import numpy as np
import dagshub
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
import optuna
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from project import logger
from project.utils.common import save_json, save_bin
from project.entity.config_entity import ModelTrainerConfig


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

        dagshub.init(repo_owner="JavithNaseem-J", repo_name="FraudGuard")
        mlflow.set_tracking_uri("https://dagshub.com/JavithNaseem-J/FraudGuard.mlflow")
        mlflow.set_experiment("Bank-Fraud-Detection")

        self.models = {
            "XGBoost": {
                "class": XGBClassifier,
                "search_space": lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 3, 10),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                },
                "mlflow_module": mlflow.xgboost,
            },
            "RandomForest": {
                "class": RandomForestClassifier,
                "search_space": lambda trial: {
                    "n_estimators": trial.suggest_int("n_estimators", 50, 300),
                    "max_depth": trial.suggest_int("max_depth", 5, 20),
                },
                "mlflow_module": mlflow.sklearn,
            },
            "LogisticRegression": {
                "class": LogisticRegression,
                "search_space": lambda trial: {
                    "C": trial.suggest_float("C", 0.01, 10.0),
                    "max_iter": trial.suggest_int("max_iter", 100, 500),
                },
                "mlflow_module": mlflow.sklearn,
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

            with mlflow.start_run(run_name=f"{model_name}_HPO", nested=False):
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
                    mean_score = scores.mean()
                    std_score = scores.std()

                    with mlflow.start_run(run_name="Trial", nested=True):
                        mlflow.log_params(params)
                        mlflow.log_metric("cv_score", mean_score)
                        mlflow.log_metric("cv_std", std_score)
                        mlflow.set_tags({
                            "model_name": model_name,
                            "trial_number": trial.number,
                            "stage": "HPO"
                        })
                    return mean_score

                study = optuna.create_study(direction="maximize")
                study.optimize(objective, n_trials=self.config.n_iter)

                best_params = study.best_params
                best_score = study.best_value
                best_model = model_info["class"](**best_params)
                cv = StratifiedKFold(n_splits=self.config.cv_folds, shuffle=True, random_state=42)
                best_scores = cross_val_score(best_model, train_x, train_y, scoring=self.config.scoring, cv=cv)
                best_std = best_scores.std()

                mlflow.log_params(best_params)
                mlflow.log_metric("best_cv_score", best_score)
                mlflow.log_metric("best_cv_std", best_std)
                mlflow.set_tag("best_model_candidate", "true")

                if best_score > best_overall["score"]:
                    best_overall.update({
                        "model_name": model_name,
                        "score": best_score,
                        "std": best_std,
                        "params": best_params
                    })

        # Train final best model and log it separately
        best_model_class = self.models[best_overall["model_name"]]["class"]
        best_model = best_model_class(**best_overall["params"])
        best_model.fit(train_x, train_y)

        with mlflow.start_run(run_name=f"{best_overall['model_name']}_final"):
            mlflow.log_params(best_overall["params"])
            mlflow.log_metric("best_cv_score", best_overall["score"])
            mlflow.log_metric("best_cv_std", best_overall["std"])
            mlflow.set_tags({"model_name": best_overall["model_name"], "stage": "final"})

            mlflow.sklearn.log_model(
                best_model,
                artifact_path="model",
                registered_model_name=f"{best_overall['model_name']}_Model"
            )

            model_path = os.path.join(self.config.root_dir, self.config.model_name)
            save_bin(data=best_model, path=Path(model_path))

            best_model_info_path = os.path.join(self.config.root_dir, "best_model_info.json")
            save_json(path=Path(best_model_info_path), data=best_overall)
            mlflow.log_artifact(best_model_info_path)

        logger.info(f"Best model overall: {best_overall}")
        return best_overall
