from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from FraudGuard import logger
from FraudGuard.cloud.settings import AppSettings
from FraudGuard.cloud.supabase import supabase_api_headers


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class PredictionRecord:
    prediction_id: str
    request_id: str
    model_version: str
    release_id: str
    transaction_amount: float | None
    score: float
    threshold: float
    decision: str
    score_is_calibrated: bool
    latency_ms: float
    created_at: str


class SupabasePersistence:
    """Server-only access to the sanitized demo persistence schema."""

    def __init__(self, settings: AppSettings):
        self.settings = settings
        self.enabled = settings.supabase_configured

    @property
    def mode(self) -> str:
        return "supabase" if self.enabled else "local_noop"

    def persist_predictions(self, records: list[PredictionRecord]) -> int:
        if not self.enabled or not records:
            return 0
        result = self._request(
            "prediction_requests", "POST", [asdict(record) for record in records]
        )
        return len(records) if result is True else 0

    def persist_model_release(
        self,
        *,
        release_id: str,
        model_version: str,
        model_name: str,
        artifact_schema_version: int,
        threshold: float,
        score_is_calibrated: bool,
        metadata: dict[str, Any],
    ) -> bool:
        if not self.enabled:
            return False
        existing = self._request(
            "model_releases",
            "GET",
            query={
                "select": "release_id",
                "release_id": f"eq.{release_id}",
                "limit": "1",
            },
        )
        if isinstance(existing, list) and existing:
            return True
        payload = {
            "release_id": release_id,
            "model_version": model_version,
            "model_name": model_name,
            "artifact_schema_version": artifact_schema_version,
            "artifact_metadata": metadata,
            "threshold": threshold,
            "false_positive_cost": metadata.get("false_positive_cost"),
            "false_negative_cost": metadata.get("false_negative_cost"),
            "score_is_calibrated": score_is_calibrated,
            "feature_schema_summary": metadata.get("feature_schema_summary", {}),
            "released_at": metadata.get("released_at", utc_now_iso()),
            "created_at": utc_now_iso(),
        }
        result = self._request("model_releases", "POST", payload)
        return result is True

    def fetch_recent_predictions(
        self, *, cutoff: str, limit: int
    ) -> list[dict[str, Any]]:
        if not self.enabled:
            return []
        selected = (
            "prediction_id,request_id,created_at,transaction_amount,score,threshold,"
            "decision,score_is_calibrated,latency_ms,model_version,release_id"
        )
        result = self._request(
            "prediction_requests",
            "GET",
            query={
                "select": selected,
                "created_at": f"gte.{cutoff}",
                "order": "created_at.desc",
                "limit": str(limit),
            },
        )
        return result if isinstance(result, list) else []

    def delete_predictions_before(self, cutoff: str) -> int | None:
        if not self.enabled:
            return None
        result = self._request(
            "prediction_requests",
            "DELETE",
            query={"created_at": f"lt.{cutoff}", "select": "prediction_id"},
            prefer="return=representation",
        )
        return len(result) if isinstance(result, list) else None

    def delete_all_predictions(self) -> int | None:
        if not self.enabled:
            return None
        result = self._request(
            "prediction_requests",
            "DELETE",
            query={"prediction_id": "not.is.null", "select": "prediction_id"},
            prefer="return=representation",
        )
        return len(result) if isinstance(result, list) else None

    def _request(
        self,
        table: str,
        method: str,
        payload: dict[str, Any] | list[dict[str, Any]] | None = None,
        *,
        query: dict[str, str] | None = None,
        prefer: str = "return=minimal",
    ) -> bool | list[dict[str, Any]]:
        query_string = urllib.parse.urlencode(query or {}, safe=".,")
        url = f"{self.settings.supabase_url}/rest/v1/{table}"
        if query_string:
            url = f"{url}?{query_string}"
        data = json.dumps(payload).encode("utf-8") if payload is not None else None
        request = urllib.request.Request(
            url=url,
            data=data,
            method=method,
            headers={
                "Content-Type": "application/json",
                **supabase_api_headers(self.settings.supabase_service_role_key),
                "Prefer": prefer,
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                if not 200 <= response.status < 300:
                    return False
                body = response.read()
                return json.loads(body) if body else True
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            logger.warning(
                "provider=supabase operation=%s table=%s outcome=failed category=%s",
                method.lower(),
                table,
                error.__class__.__name__,
            )
            return False
