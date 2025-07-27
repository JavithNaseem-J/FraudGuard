import os
import json
import joblib
import mlflow
import dagshub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

from FraudGuard import logger
from FraudGuard.utils.helpers import save_json
from FraudGuard.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        os.environ["MLFLOW_TRACKING_USERNAME"] = self.config.mlflow_username
        os.environ["MLFLOW_TRACKING_PASSWORD"] = self.config.mlflow_password

        dagshub.init(repo_owner='JavithNaseem-J', repo_name='Condition2Cure')
        mlflow.set_tracking_uri('https://dagshub.com/JavithNaseem-J/FraudGuard.mlflow')
        mlflow.set_experiment("Fraud-Detection")

    def evaluation(self):
        if not os.path.exists(self.config.test_path):
            raise FileNotFoundError(f"Test data not found: {self.config.test_path}")
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")
        if not os.path.exists(self.config.preprocess_path):
            raise FileNotFoundError(f"Preprocessor not found: {self.config.preprocess_path}")

        test_df = pd.read_csv(self.config.test_path)
        model = joblib.load(self.config.model_path)
        preprocessor = joblib.load(self.config.preprocess_path)

        target_column = self.config.target_column
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        X_test_transformed = preprocessor.transform(X_test)

        preds = model.predict(X_test_transformed)
        proba = model.predict_proba(X_test_transformed)[:, 1] if hasattr(model, "predict_proba") else None

        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision_weighted": precision_score(y_test, preds, average="weighted"),
            "recall_weighted": recall_score(y_test, preds, average="weighted"),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
            "precision_macro": precision_score(y_test, preds, average="macro"),
            "recall_macro": recall_score(y_test, preds, average="macro"),
            "f1_macro": f1_score(y_test, preds, average="macro"),
        }

        if proba is not None:
            fpr, tpr, _ = roc_curve(y_test, proba)
            metrics["auc"] = auc(fpr, tpr)

        os.makedirs(self.config.root_dir, exist_ok=True)
        save_json(path=Path(self.config.metrics_path), data=metrics)

        
        with mlflow.start_run(run_name="Model Evaluation"):
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.set_tag("stage", "evaluation")

            # Log artifacts
            mlflow.log_artifact(self.config.metrics_path)
            mlflow.log_artifact(self.config.model_path)
            mlflow.log_artifact(self.config.preprocess_path)

            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(self.config.cm_path)
            plt.close()
            mlflow.log_artifact(self.config.cm_path)

            # ROC Curve
            if proba is not None:
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.2f}")
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(self.config.roc_path, bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(self.config.roc_path)


        logger.info("Model evaluation complete. Metrics and plots logged.")
        return metrics
