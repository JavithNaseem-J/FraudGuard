import os
import json
import joblib
import mlflow
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix

from FraudGuard import logger
from FraudGuard.utils.helpers import save_json, init_mlflow_tracking
from FraudGuard.entity.config_entity import ModelEvaluationConfig


class Evaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

        # Initialize MLflow tracking (centralized, idempotent)
        init_mlflow_tracking(
            mlflow_username=self.config.mlflow_username,
            mlflow_password=self.config.mlflow_password
        )


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

            # SHAP Feature Importance (Model Interpretability)
            self._generate_shap_plots(model, X_test_transformed, X_test.columns)

        logger.info("Model evaluation complete. Metrics and plots logged.")
        return metrics

    def _generate_shap_plots(self, model, X_test, feature_names):
        """Generate SHAP plots for model interpretability."""
        try:
            logger.info("Generating SHAP feature importance plots...")
            
            # Use TreeExplainer for tree-based models (XGBoost, LightGBM, CatBoost)
            explainer = shap.TreeExplainer(model)
            
            # Calculate SHAP values (use sample for speed)
            sample_size = min(500, len(X_test))
            X_sample = X_test[:sample_size]
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different output formats
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # For binary classification, use class 1
            
            # Create DataFrame with feature names
            X_sample_df = pd.DataFrame(X_sample, columns=feature_names)
            
            # Summary Plot (Bar)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample_df, plot_type="bar", show=False)
            plt.title("SHAP Feature Importance")
            plt.tight_layout()
            shap_bar_path = os.path.join(self.config.root_dir, "shap_importance.png")
            plt.savefig(shap_bar_path, bbox_inches="tight", dpi=150)
            plt.close()
            mlflow.log_artifact(shap_bar_path)
            
            # Summary Plot (Beeswarm)
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, X_sample_df, show=False)
            plt.title("SHAP Summary Plot")
            plt.tight_layout()
            shap_summary_path = os.path.join(self.config.root_dir, "shap_summary.png")
            plt.savefig(shap_summary_path, bbox_inches="tight", dpi=150)
            plt.close()
            mlflow.log_artifact(shap_summary_path)
            
            logger.info("SHAP plots generated successfully.")
            
        except Exception as e:
            logger.warning(f"Could not generate SHAP plots: {str(e)}")