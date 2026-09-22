from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class AppSettings:
    app_env: str
    model_mode: str
    model_artifact_root: Path
    transaction_candidate_artifact_root: Path
    transaction_artifact_release_id: str
    artifact_storage_bucket: str
    artifact_cache_root: Path
    artifact_download_timeout_seconds: int
    artifact_download_retries: int
    rollback_release_id: str
    public_base_url: str
    port: int
    supabase_url: str
    supabase_service_role_key: str
    upstash_redis_rest_url: str
    upstash_redis_rest_token: str
    upstash_fail_closed: bool
    rate_limit_requests: int
    rate_limit_window_seconds: int
    prediction_persistence_enabled: bool
    auth_required: bool
    fraudguard_api_key: str
    max_request_bytes: int
    max_batch_rows: int

    @property
    def supabase_configured(self) -> bool:
        return bool(self.supabase_url and self.supabase_service_role_key)

    @property
    def upstash_configured(self) -> bool:
        return bool(self.upstash_redis_rest_url and self.upstash_redis_rest_token)

    @property
    def transaction_candidate_enabled(self) -> bool:
        return self.model_mode == "transaction_candidate"


def load_settings() -> AppSettings:
    project_root = Path(__file__).resolve().parents[3]
    model_mode = os.getenv("FRAUD_MODEL_MODE", "transaction_candidate").strip().lower()
    if model_mode not in {"baseline", "transaction_candidate"}:
        model_mode = "transaction_candidate"
    return AppSettings(
        app_env=os.getenv("APP_ENV", "local"),
        model_mode=model_mode,
        model_artifact_root=Path(
            os.getenv(
                "MODEL_ARTIFACT_ROOT", str(project_root / "artifacts" / "trainer")
            )
        ),
        transaction_candidate_artifact_root=Path(
            os.getenv(
                "TRANSACTION_CANDIDATE_ARTIFACT_ROOT",
                str(
                    project_root
                    / "artifacts"
                    / "benchmark"
                    / "transaction_data"
                    / "candidate"
                ),
            )
        ),
        transaction_artifact_release_id=os.getenv(
            "TRANSACTION_ARTIFACT_RELEASE_ID", ""
        ).strip(),
        artifact_storage_bucket=os.getenv("ARTIFACT_STORAGE_BUCKET", "").strip(),
        artifact_cache_root=Path(
            os.getenv(
                "ARTIFACT_CACHE_ROOT",
                str(project_root / "artifacts" / "releases"),
            )
        ),
        artifact_download_timeout_seconds=int(
            os.getenv("ARTIFACT_DOWNLOAD_TIMEOUT_SECONDS", "20")
        ),
        artifact_download_retries=int(os.getenv("ARTIFACT_DOWNLOAD_RETRIES", "2")),
        rollback_release_id=os.getenv("ROLLBACK_RELEASE_ID", "").strip(),
        public_base_url=os.getenv("PUBLIC_BASE_URL", "http://localhost:8000"),
        port=int(os.getenv("PORT", "8000")),
        supabase_url=os.getenv("SUPABASE_URL", "").rstrip("/"),
        supabase_service_role_key=os.getenv("SUPABASE_SERVICE_ROLE_KEY", ""),
        upstash_redis_rest_url=os.getenv("UPSTASH_REDIS_REST_URL", "").rstrip("/"),
        upstash_redis_rest_token=os.getenv("UPSTASH_REDIS_REST_TOKEN", ""),
        upstash_fail_closed=_truthy(os.getenv("UPSTASH_FAIL_CLOSED")),
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "20")),
        rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
        prediction_persistence_enabled=not _truthy(
            os.getenv("DISABLE_PREDICTION_PERSISTENCE")
        ),
        auth_required=_truthy(os.getenv("AUTH_REQUIRED")),
        fraudguard_api_key=os.getenv("FRAUDGUARD_API_KEY", ""),
        max_request_bytes=int(os.getenv("MAX_REQUEST_BYTES", "1048576")),
        max_batch_rows=int(os.getenv("MAX_BATCH_ROWS", "100")),
    )
