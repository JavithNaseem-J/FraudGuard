from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from FraudGuard import logger
from FraudGuard.cloud.settings import AppSettings


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    request_id: str
    model_version: str
    model_mode: str
    score: float
    threshold: float
    decision: str
    score_is_calibrated: bool
    latency_ms: float
    metadata: dict[str, Any]


class SupabasePersistence:
    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.enabled = (
            settings.prediction_persistence_enabled and settings.supabase_configured
        )

    @property
    def mode(self) -> str:
        return "supabase" if self.enabled else "local_noop"

    def persist_prediction(self, record: PredictionRecord) -> bool:
        if not self.enabled:
            return False

        payload = {
            "prediction_id": record.prediction_id,
            "request_id": record.request_id,
            "model_version": record.model_version,
            "model_mode": record.model_mode,
            "score": record.score,
            "threshold": record.threshold,
            "decision": record.decision,
            "score_is_calibrated": record.score_is_calibrated,
            "latency_ms": record.latency_ms,
            "metadata": record.metadata,
            "created_at": utc_now_iso(),
        }
        return self._insert("prediction_requests", payload)

    def persist_model_release(
        self,
        *,
        model_version: str,
        model_name: str,
        artifact_schema_version: int,
        threshold: float,
        score_is_calibrated: bool,
        metadata: dict[str, Any],
    ) -> bool:
        if not self.enabled:
            return False

        payload = {
            "release_id": metadata.get("release_id"),
            "model_version": model_version,
            "model_name": model_name,
            "artifact_schema_version": artifact_schema_version,
            "artifact_metadata": metadata,
            "threshold": threshold,
            "false_positive_cost": metadata.get("false_positive_cost"),
            "false_negative_cost": metadata.get("false_negative_cost"),
            "score_is_calibrated": score_is_calibrated,
            "model_mode": metadata.get("model_mode", "baseline"),
            "feature_schema_summary": metadata.get("feature_schema_summary", {}),
            "released_at": metadata.get("released_at", utc_now_iso()),
            "created_at": utc_now_iso(),
        }
        return self._insert("model_releases", payload)

    def persist_audit_event(
        self, event_type: str, entity_id: str | None, metadata: dict[str, Any]
    ) -> bool:
        if not self.enabled:
            return False

        payload = {
            "event_type": event_type,
            "entity_id": entity_id,
            "metadata": metadata,
            "created_at": utc_now_iso(),
        }
        return self._insert("audit_events", payload)

    def persist_feedback(
        self,
        prediction_id: str,
        confirmed_label: int,
        reviewer_decision: str | None,
        feedback_source: str,
        metadata: dict[str, Any],
    ) -> bool:
        if not self.enabled:
            return False

        payload = {
            "prediction_id": prediction_id,
            "confirmed_label": confirmed_label,
            "reviewer_decision": reviewer_decision,
            "feedback_source": feedback_source,
            "metadata": metadata,
            "created_at": utc_now_iso(),
        }
        return self._insert("prediction_feedback", payload)

    def _insert(self, table: str, payload: dict[str, Any]) -> bool:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            url=f"{self.settings.supabase_url}/rest/v1/{table}",
            data=body,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "apikey": self.settings.supabase_service_role_key,
                "Authorization": f"Bearer {self.settings.supabase_service_role_key}",
                "Prefer": "return=minimal",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return 200 <= response.status < 300
        except (urllib.error.URLError, TimeoutError) as error:
            logger.warning("Supabase insert failed for %s: %s", table, error)
            return False
