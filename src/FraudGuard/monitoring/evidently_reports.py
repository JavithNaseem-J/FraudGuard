from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

APPROVED_MONITORING_COLUMNS = (
    "created_at",
    "transaction_amount",
    "score",
    "threshold",
    "decision",
    "latency_ms",
    "model_version",
    "release_id",
)
NUMERIC_MONITORING_COLUMNS = (
    "transaction_amount",
    "score",
    "threshold",
    "latency_ms",
)


def sanitize_monitoring_frame(frame: pd.DataFrame) -> pd.DataFrame:
    allowed = [column for column in APPROVED_MONITORING_COLUMNS if column in frame]
    sanitized = frame.loc[:, allowed].copy()
    for column in NUMERIC_MONITORING_COLUMNS:
        if column in sanitized:
            sanitized[column] = pd.to_numeric(sanitized[column], errors="coerce")
    if "decision" in sanitized:
        sanitized["decision"] = sanitized["decision"].astype("string")
    if "created_at" in sanitized:
        sanitized["created_at"] = pd.to_datetime(
            sanitized["created_at"], errors="coerce", utc=True
        )
    return sanitized


def _window(frame: pd.DataFrame) -> dict[str, str | None]:
    if "created_at" not in frame or frame["created_at"].dropna().empty:
        return {"start": None, "end": None}
    return {
        "start": frame["created_at"].min().isoformat(),
        "end": frame["created_at"].max().isoformat(),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def generate_output_monitoring_report(
    reference_path: Path,
    current_path: Path,
    output_path: Path,
    *,
    metrics_path: Path,
    release_id: str = "local",
    min_rows: int = 25,
) -> dict[str, Any]:
    reference = sanitize_monitoring_frame(pd.read_csv(reference_path))
    current = sanitize_monitoring_frame(pd.read_csv(current_path))
    metrics: dict[str, Any] = {
        "run_type": "sanitized_output_drift",
        "release_id": release_id,
        "reference_rows": int(len(reference)),
        "current_rows": int(len(current)),
        "reference_window": _window(reference),
        "current_window": _window(current),
        "columns": list(current.columns),
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }
    if reference.empty or current.empty:
        metrics["status"] = "no_data"
        _write_json(metrics_path, metrics)
        return metrics
    if len(reference) < min_rows or len(current) < min_rows:
        metrics["status"] = "insufficient_data"
        _write_json(metrics_path, metrics)
        return metrics

    try:
        from evidently import Report
        from evidently.presets import DataDriftPreset
    except ImportError as error:
        raise RuntimeError(
            "Install the development dependencies to generate Evidently reports"
        ) from error

    comparison_columns = [
        column
        for column in (
            "transaction_amount",
            "score",
            "threshold",
            "decision",
            "latency_ms",
        )
        if column in reference and column in current
    ]
    if not comparison_columns:
        metrics["status"] = "no_data"
        metrics["reason"] = "no shared approved monitoring columns"
        _write_json(metrics_path, metrics)
        return metrics

    report = Report([DataDriftPreset()])
    result = report.run(
        reference_data=reference.loc[:, comparison_columns],
        current_data=current.loc[:, comparison_columns],
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.save_html(str(output_path))
    metrics.update({"status": "success", "report_path": str(output_path)})
    _write_json(metrics_path, metrics)
    return metrics
