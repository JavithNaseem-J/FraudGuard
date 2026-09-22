from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
)

from FraudGuard.utils.costs import cost_weighted_loss


APPROVED_MONITORING_COLUMNS = {
    "prediction_id",
    "model_version",
    "release_id",
    "score",
    "threshold",
    "decision",
    "created_at",
    "confirmed_label",
}


def sanitize_monitoring_frame(frame: pd.DataFrame) -> pd.DataFrame:
    allowed = [
        column for column in frame.columns if column in APPROVED_MONITORING_COLUMNS
    ]
    sanitized = frame.loc[:, allowed].copy()
    if "score" in sanitized:
        sanitized["score"] = pd.to_numeric(sanitized["score"], errors="coerce")
    if "threshold" in sanitized:
        sanitized["threshold"] = pd.to_numeric(sanitized["threshold"], errors="coerce")
    if "confirmed_label" in sanitized:
        sanitized["confirmed_label"] = pd.to_numeric(
            sanitized["confirmed_label"], errors="coerce"
        )
    return sanitized


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def generate_unlabeled_drift_report(
    reference_path: Path,
    current_path: Path,
    output_path: Path,
    *,
    metrics_path: Path | None = None,
    release_id: str = "local",
    reference_version: str = "local-reference",
    min_rows: int = 25,
) -> Path:
    """Generate an Evidently HTML drift report when Evidently is installed."""
    reference = sanitize_monitoring_frame(pd.read_csv(reference_path))
    current = sanitize_monitoring_frame(pd.read_csv(current_path))
    metrics = {
        "run_type": "drift",
        "release_id": release_id,
        "reference_version": reference_version,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    if len(reference) == 0 or len(current) == 0:
        metrics["status"] = "no_data"
        if metrics_path:
            _write_json(metrics_path, metrics)
        return output_path
    if len(reference) < min_rows or len(current) < min_rows:
        metrics["status"] = "insufficient_data"
        if metrics_path:
            _write_json(metrics_path, metrics)
        return output_path

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as error:
        raise RuntimeError(
            "Evidently is not installed. Install evidently to generate drift reports."
        ) from error

    report = Report([DataDriftPreset()])
    result = report.run(reference_data=reference, current_data=current)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(output_path))
    metrics["status"] = "success"
    metrics["report_path"] = str(output_path)
    if metrics_path:
        _write_json(metrics_path, metrics)
    return output_path


def generate_delayed_label_performance_frame(
    predictions_path: Path,
    feedback_path: Path,
    prediction_id_column: str = "prediction_id",
) -> pd.DataFrame:
    predictions = pd.read_csv(predictions_path)
    feedback = pd.read_csv(feedback_path)
    return sanitize_monitoring_frame(
        predictions.merge(feedback, on=prediction_id_column, how="inner")
    )


def generate_delayed_label_performance_report(
    predictions_path: Path,
    feedback_path: Path,
    metrics_path: Path,
    *,
    prediction_id_column: str = "prediction_id",
    false_positive_cost: float = 1.0,
    false_negative_cost: float = 20.0,
) -> dict[str, Any]:
    predictions = sanitize_monitoring_frame(pd.read_csv(predictions_path))
    joined = generate_delayed_label_performance_frame(
        predictions_path,
        feedback_path,
        prediction_id_column=prediction_id_column,
    )
    label_count = (
        int(joined["confirmed_label"].notna().sum())
        if "confirmed_label" in joined
        else 0
    )
    metrics: dict[str, Any] = {
        "run_type": "delayed_label_performance",
        "status": "success" if label_count else "no_data",
        "prediction_rows": int(len(predictions)),
        "label_count": label_count,
        "label_coverage": (
            float(label_count / len(predictions)) if len(predictions) else 0.0
        ),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    if not label_count:
        _write_json(metrics_path, metrics)
        return metrics

    labeled = joined.dropna(subset=["confirmed_label", "score", "threshold"]).copy()
    y_true = labeled["confirmed_label"].astype(int).to_numpy()
    scores = labeled["score"].astype(float).to_numpy()
    threshold = float(labeled["threshold"].median())
    y_pred = (scores >= threshold).astype(int)
    metrics.update(
        {
            "threshold": threshold,
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "brier_score": float(brier_score_loss(y_true, scores)),
            "cost_weighted": cost_weighted_loss(
                y_true,
                y_pred,
                false_positive_cost=false_positive_cost,
                false_negative_cost=false_negative_cost,
            ),
        }
    )
    if len(np.unique(y_true)) == 2:
        metrics["average_precision"] = float(average_precision_score(y_true, scores))
    else:
        metrics["average_precision"] = None
        metrics["average_precision_note"] = (
            "skipped because only one label class is present"
        )
    _write_json(metrics_path, metrics)
    return metrics
