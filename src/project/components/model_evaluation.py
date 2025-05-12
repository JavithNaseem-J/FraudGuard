# model_evaluation.py
import os
import json
import joblib
import mlflow
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, confusion_matrix
)
from project import logger
from project.utils.common import save_json
from project.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluation(self):
        # Validate input paths
        if not os.path.exists(self.config.test_path):
            raise FileNotFoundError(f"Test data not found: {self.config.test_path}")
        if not os.path.exists(self.config.model_path):
            raise FileNotFoundError(f"Model not found: {self.config.model_path}")
        if not os.path.exists(self.config.preprocess_path):
            raise FileNotFoundError(f"Preprocessor not found: {self.config.preprocess_path}")

        # Load data and components
        test_df = pd.read_csv(self.config.test_path)
        model = joblib.load(self.config.model_path)
        preprocessor = joblib.load(self.config.preprocess_path)

        target_column = self.config.target_column
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]
        X_test_transformed = preprocessor.transform(X_test)

        # Predictions
        preds = model.predict(X_test_transformed)
        proba = model.predict_proba(X_test_transformed)[:, 1] if hasattr(model, "predict_proba") else None

        # Metrics
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision_weighted": precision_score(y_test, preds, average="weighted"),
            "recall_weighted": recall_score(y_test, preds, average="weighted"),
            "f1_weighted": f1_score(y_test, preds, average="weighted"),
        }

        if proba is not None:
            fpr, tpr, _ = roc_curve(y_test, proba)
            metrics["auc"] = auc(fpr, tpr)

        # Save metrics JSON
        os.makedirs(self.config.root_dir, exist_ok=True)
        save_json(path=Path(self.config.metrics_path), data=metrics)

        with mlflow.start_run(run_name="Model Evaluation"):
            mlflow.log_metrics(metrics)
            mlflow.set_tag("stage", "evaluation")
            mlflow.log_artifact(Path(self.config.metrics_path))

            # Confusion Matrix
            cm = confusion_matrix(y_test, preds)
            plt.figure(figsize=(6, 4))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
            plt.title("Confusion Matrix")
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            plt.tight_layout()
            plt.savefig(Path(self.config.cm_path))
            plt.close()
            mlflow.log_artifact(Path(self.config.cm_path))

            # ROC Curve (if probability available)
            if proba is not None:
                plt.figure(figsize=(6, 4))
                plt.plot(fpr, tpr, label=f"AUC = {metrics['auc']:.2f}")
                plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
                plt.xlabel("False Positive Rate")
                plt.ylabel("True Positive Rate")
                plt.title("ROC Curve")
                plt.legend()
                plt.tight_layout()
                plt.savefig(Path(self.config.roc_path), bbox_inches="tight")
                plt.close()
                mlflow.log_artifact(Path(self.config.roc_path))

        logger.info("Model evaluation complete. Metrics and plots logged.")
        return metrics
