from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from FraudGuard.cloud.artifacts import (  # noqa: E402
    build_transaction_release_manifest,
    validate_release_directory,
    write_manifest,
)
from FraudGuard.cloud.settings import load_settings  # noqa: E402


def _present(value: str) -> str:
    return "set" if value else "missing"


def _check_files(label: str, paths: list) -> list[str]:
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        print(f"Missing {label} artifacts:")
        for name in missing:
            print(f"- {name}")
    else:
        print(f"{label} artifacts are present.")
    return missing


def _validate_local_transaction_release(settings) -> list[str]:
    failures: list[str] = []
    required = [
        settings.transaction_candidate_artifact_root / "model.joblib",
        settings.transaction_candidate_artifact_root / "threshold.json",
        settings.transaction_candidate_artifact_root / "metadata.json",
        settings.transaction_candidate_artifact_root / "feature_audit.json",
    ]
    failures.extend(_check_files("transaction model", required))
    if failures:
        return failures
    manifest_path = settings.transaction_candidate_artifact_root / "manifest.json"
    release_id = settings.transaction_artifact_release_id or "local-predeploy"
    if not manifest_path.exists():
        manifest = build_transaction_release_manifest(
            settings.transaction_candidate_artifact_root,
            release_id=release_id,
        )
        write_manifest(settings.transaction_candidate_artifact_root, manifest)
    try:
        validate_release_directory(
            settings.transaction_candidate_artifact_root,
            expected_release_id=(
                release_id if settings.transaction_artifact_release_id else None
            ),
        )
    except Exception as error:
        failures.append(
            f"Transaction release validation failed: {error.__class__.__name__}"
        )
    return failures


def main() -> int:
    settings = load_settings()
    failures: list[str] = []
    print(f"APP_ENV={settings.app_env}")
    print(f"FRAUD_MODEL_MODE={settings.model_mode}")
    print(f"AUTH_REQUIRED={settings.auth_required}")
    print(f"FRAUDGUARD_API_KEY={_present(settings.fraudguard_api_key)}")
    print(f"MAX_REQUEST_BYTES={settings.max_request_bytes}")
    print(f"MAX_BATCH_ROWS={settings.max_batch_rows}")
    print(f"SUPABASE_URL={_present(settings.supabase_url)}")
    print(f"SUPABASE_SERVICE_ROLE_KEY={_present(settings.supabase_service_role_key)}")
    print(f"UPSTASH_REDIS_REST_URL={_present(settings.upstash_redis_rest_url)}")
    print(f"UPSTASH_REDIS_REST_TOKEN={_present(settings.upstash_redis_rest_token)}")
    print(
        f"TRANSACTION_ARTIFACT_RELEASE_ID={_present(settings.transaction_artifact_release_id)}"
    )
    print(f"ARTIFACT_STORAGE_BUCKET={_present(settings.artifact_storage_bucket)}")
    print(f"ARTIFACT_CACHE_ROOT={settings.artifact_cache_root}")

    if settings.app_env in {"render", "staging", "production"}:
        if not settings.auth_required:
            failures.append("AUTH_REQUIRED should be true for public deployments")
        if not settings.fraudguard_api_key:
            failures.append("FRAUDGUARD_API_KEY is required when auth is enabled")
        if not settings.upstash_configured:
            failures.append(
                "Upstash REST URL and token are required for public rate limiting"
            )
        if settings.transaction_candidate_enabled:
            if not settings.transaction_artifact_release_id:
                failures.append(
                    "TRANSACTION_ARTIFACT_RELEASE_ID is required for public transaction deployments"
                )
            if not settings.artifact_storage_bucket:
                failures.append(
                    "ARTIFACT_STORAGE_BUCKET is required for public transaction deployments"
                )
            if not settings.supabase_configured:
                failures.append(
                    "Supabase URL and service role key are required for artifact delivery"
                )
    if settings.max_request_bytes <= 0:
        failures.append("MAX_REQUEST_BYTES must be positive")
    if settings.max_batch_rows <= 0:
        failures.append("MAX_BATCH_ROWS must be positive")

    print(
        "TRANSACTION_CANDIDATE_ARTIFACT_ROOT="
        f"{settings.transaction_candidate_artifact_root}"
    )
    local_release_available = (
        settings.transaction_candidate_artifact_root / "model.joblib"
    ).exists()
    if local_release_available:
        failures.extend(_validate_local_transaction_release(settings))
    else:
        print(
            "No local transaction artifacts found; remote release configuration will be used at startup."
        )
    if failures:
        print("Predeploy check failed.")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print("Predeploy check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
