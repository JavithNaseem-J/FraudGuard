from __future__ import annotations

import os
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from FraudGuard import logger
from FraudGuard.entity.config_entity import ModelEvaluationConfig
from FraudGuard.utils.costs import cost_weighted_loss
from FraudGuard.utils.helpers import init_mlflow_tracking, load_json, save_json


class Evaluation:
    def __init__(self, config: ModelEvaluationConfig):
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

    def _load_contract(self) -> tuple[object, dict, dict]:
        required_paths = [
            self.config.test_path,
            self.config.model_path,
            self.config.threshold_path,
            self.config.model_version_path,
        ]
        missing_paths = [
            str(path) for path in required_paths if not Path(path).exists()
        ]
        if missing_paths:
            raise FileNotFoundError(
                f"Required evaluation artifacts missing: {missing_paths}"
            )

        model = joblib.load(self.config.model_path)
        threshold_info = load_json(Path(self.config.threshold_path))
        metadata = load_json(Path(self.config.model_version_path))

        if metadata.get("artifact_schema_version") != 2:
            raise ValueError("Unsupported or missing model artifact schema version")
        threshold = float(threshold_info.get("optimal_threshold", -1))
        if not 0 <= threshold <= 1:
            raise ValueError(f"Invalid threshold: {threshold}")
        if not np.isclose(threshold, float(metadata.get("optimal_threshold", -2))):
            raise ValueError("Threshold artifact does not match model metadata")

        return model, threshold_info, metadata

    def evaluation(self):
        model, threshold_info, metadata = self._load_contract()
        test_df = pd.read_csv(self.config.test_path)
        target_column = self.config.target_column
        if target_column not in test_df.columns:
            raise ValueError(f"Missing test target column: {target_column}")

        feature_names = metadata["feature_names"]
        missing_features = sorted(set(feature_names) - set(test_df.columns))
        if missing_features:
            raise ValueError(f"Test data is missing features: {missing_features}")

        X_test = test_df[feature_names]
        y_test = test_df[target_column].astype(int)
        probabilities = model.predict_proba(X_test)[:, 1]
        threshold = float(threshold_info["optimal_threshold"])
        predictions = (probabilities >= threshold).astype(int)

        matrix = confusion_matrix(y_test, predictions, labels=[0, 1])
        false_positive_cost = float(
            metadata.get("false_positive_cost", self.config.false_positive_cost)
        )
        false_negative_cost = float(
            metadata.get("false_negative_cost", self.config.false_negative_cost)
        )
        cost_metrics = cost_weighted_loss(
            y_test.to_numpy(),
            predictions,
            false_positive_cost=false_positive_cost,
            false_negative_cost=false_negative_cost,
        )
        metrics = {
            "test_rows": int(len(test_df)),
            "positive_support": int(y_test.sum()),
            "negative_support": int((y_test == 0).sum()),
            "prevalence": float(y_test.mean()),
            "threshold": threshold,
            "threshold_objective": threshold_info.get("objective", "unknown"),
            "precision": float(precision_score(y_test, predictions, zero_division=0)),
            "recall": float(recall_score(y_test, predictions, zero_division=0)),
            "f1": float(f1_score(y_test, predictions, zero_division=0)),
            "accuracy": float(accuracy_score(y_test, predictions)),
            "average_precision": float(average_precision_score(y_test, probabilities)),
            "roc_auc": float(roc_auc_score(y_test, probabilities)),
            "brier_score": float(brier_score_loss(y_test, probabilities)),
            "baseline_average_precision": float(y_test.mean()),
            "baseline_cv_average_precision": float(metadata["baseline_cv_score"]),
            "score_is_calibrated": bool(metadata["score_is_calibrated"]),
            "confusion_matrix": matrix.tolist(),
            "cost_weighted": cost_metrics,
        }

        os.makedirs(self.config.root_dir, exist_ok=True)
        save_json(path=Path(self.config.metrics_path), data=metrics)
        self._save_confusion_matrix(matrix)
        self._save_roc_curve(y_test, probabilities, metrics["roc_auc"])
        self._save_pr_curve(y_test, probabilities, metrics["average_precision"])
        self._generate_shap_plots(model, X_test)

        logger.info("Model evaluation complete: %s", metrics)
        return metrics

    def _save_confusion_matrix(self, matrix: np.ndarray) -> None:
        figure, axis = plt.subplots(figsize=(6, 4))
        image = axis.imshow(matrix, interpolation="nearest", cmap="Blues")
        figure.colorbar(image, ax=axis)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    str(matrix[row_index, column_index]),
                    ha="center",
                    va="center",
                )
        axis.set(
            title="Confusion Matrix at Stored Threshold",
            xlabel="Predicted",
            ylabel="Actual",
            xticks=[0, 1],
            yticks=[0, 1],
        )
        figure.tight_layout()
        figure.savefig(self.config.cm_path)
        plt.close(figure)

    def _save_roc_curve(
        self, y_true: pd.Series, probabilities: np.ndarray, roc_auc: float
    ) -> None:
        false_positive_rate, true_positive_rate, _ = roc_curve(y_true, probabilities)
        plt.figure(figsize=(6, 4))
        plt.plot(false_positive_rate, true_positive_rate, label=f"AUC = {roc_auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.roc_path, bbox_inches="tight")
        plt.close()

    def _save_pr_curve(
        self, y_true: pd.Series, probabilities: np.ndarray, average_precision: float
    ) -> None:
        precision, recall, _ = precision_recall_curve(y_true, probabilities)
        plt.figure(figsize=(6, 4))
        plt.plot(recall, precision, label=f"AP = {average_precision:.3f}")
        plt.axhline(
            float(y_true.mean()), linestyle="--", color="gray", label="prevalence"
        )
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(self.config.pr_path, bbox_inches="tight")
        plt.close()

    def _generate_shap_plots(self, pipeline, X_test: pd.DataFrame) -> None:
        """Generate tree SHAP plots when the selected classifier supports them."""
        try:
            import shap

            preprocessor = pipeline.named_steps["preprocessor"]
            classifier = pipeline.named_steps["classifier"]
            transformed = preprocessor.transform(X_test.iloc[:500])
            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()
            feature_names = preprocessor.get_feature_names_out()
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]
            sample = pd.DataFrame(transformed, columns=feature_names)

            shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
            plt.tight_layout()
            plt.savefig(
                Path(self.config.root_dir) / "shap_importance.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()

            shap.summary_plot(shap_values, sample, show=False)
            plt.tight_layout()
            plt.savefig(
                Path(self.config.root_dir) / "shap_summary.png",
                bbox_inches="tight",
                dpi=150,
            )
            plt.close()
        except Exception as error:
            logger.warning("SHAP plots were not generated: %s", error)


if __name__ == "__main__":
    from FraudGuard.config.config import ConfigurationManager

    logger.info(">>>>>> Stage: Evaluation started <<<<<<")
    config = ConfigurationManager()
    evaluation = Evaluation(config=config.get_model_evaluation_config())
    evaluation.evaluation()
    logger.info(">>>>>> Stage: Evaluation completed <<<<<<")
