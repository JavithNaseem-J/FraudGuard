from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from FraudGuard.cloud.settings import AppSettings
from FraudGuard.pipeline.transaction_candidate_pipeline import (
    TransactionCandidatePipeline,
)

MANIFEST_FILENAME = "manifest.json"
REQUIRED_TRANSACTION_FILES = (
    "model.joblib",
    "threshold.json",
    "metadata.json",
    "feature_audit.json",
)


@dataclass(frozen=True)
class ArtifactFile:
    path: str
    size_bytes: int
    sha256: str


@dataclass(frozen=True)
class ArtifactManifest:
    release_id: str
    artifact_schema_version: int
    model_version: str
    created_at_utc: str
    files: list[ArtifactFile]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "release_id": self.release_id,
            "artifact_schema_version": self.artifact_schema_version,
            "model_version": self.model_version,
            "created_at_utc": self.created_at_utc,
            "files": [file.__dict__ for file in self.files],
            "metadata": self.metadata,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative_path(value: str) -> Path:
    candidate = Path(value)
    if candidate.is_absolute() or ".." in candidate.parts or value.strip() != value:
        raise ValueError(f"Unsafe artifact path in manifest: {value!r}")
    if not value or value.replace("\\", "/") != value:
        raise ValueError(f"Artifact paths must be relative POSIX paths: {value!r}")
    return candidate


def build_transaction_release_manifest(
    artifact_root: Path,
    *,
    release_id: str,
    model_version: str | None = None,
) -> ArtifactManifest:
    metadata_path = artifact_root / "metadata.json"
    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    resolved_model_version = (
        model_version
        or metadata.get("created_at_utc")
        or metadata.get("model_version")
        or release_id
    )
    files: list[ArtifactFile] = []
    for name in REQUIRED_TRANSACTION_FILES:
        path = artifact_root / name
        if not path.exists():
            raise FileNotFoundError(f"Missing transaction artifact: {name}")
        files.append(
            ArtifactFile(
                path=name,
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
            )
        )
    return ArtifactManifest(
        release_id=release_id,
        artifact_schema_version=1,
        model_version=str(resolved_model_version),
        created_at_utc=datetime.now(UTC).isoformat(),
        files=files,
        metadata={
            "model_mode": "transaction_candidate",
            "required_files": list(REQUIRED_TRANSACTION_FILES),
        },
    )


def write_manifest(artifact_root: Path, manifest: ArtifactManifest) -> Path:
    path = artifact_root / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return path


def load_manifest(artifact_root: Path) -> ArtifactManifest:
    path = artifact_root / MANIFEST_FILENAME
    payload = json.loads(path.read_text(encoding="utf-8"))
    files = [
        ArtifactFile(
            path=str(item["path"]),
            size_bytes=int(item["size_bytes"]),
            sha256=str(item["sha256"]),
        )
        for item in payload.get("files", [])
    ]
    return ArtifactManifest(
        release_id=str(payload["release_id"]),
        artifact_schema_version=int(payload["artifact_schema_version"]),
        model_version=str(payload["model_version"]),
        created_at_utc=str(payload["created_at_utc"]),
        files=files,
        metadata=dict(payload.get("metadata", {})),
    )


def validate_manifest(manifest: ArtifactManifest) -> None:
    if manifest.artifact_schema_version != 1:
        raise ValueError("Unsupported artifact manifest schema version")
    declared = set()
    for item in manifest.files:
        _safe_relative_path(item.path)
        declared.add(item.path)
    missing = sorted(set(REQUIRED_TRANSACTION_FILES) - declared)
    if missing:
        raise ValueError(f"Manifest is missing required files: {missing}")
    for item in manifest.files:
        if item.size_bytes <= 0:
            raise ValueError(f"Artifact size must be positive: {item.path}")
        if len(item.sha256) != 64:
            raise ValueError(f"Artifact checksum is invalid: {item.path}")


def validate_release_directory(
    artifact_root: Path,
    *,
    expected_release_id: str | None = None,
    validate_model: bool = True,
) -> ArtifactManifest:
    manifest = load_manifest(artifact_root)
    validate_manifest(manifest)
    if expected_release_id and manifest.release_id != expected_release_id:
        raise ValueError("Cached artifact release does not match selected release")
    for item in manifest.files:
        relative = _safe_relative_path(item.path)
        path = artifact_root / relative
        if not path.exists():
            raise FileNotFoundError(f"Missing release artifact: {item.path}")
        if path.stat().st_size != item.size_bytes:
            raise ValueError(f"Artifact size mismatch: {item.path}")
        if _sha256(path) != item.sha256:
            raise ValueError(f"Artifact checksum mismatch: {item.path}")
    if validate_model:
        TransactionCandidatePipeline(artifact_root)
    return manifest


def _storage_url(settings: AppSettings, release_id: str, file_name: str) -> str:
    bucket = settings.artifact_storage_bucket
    return (
        f"{settings.supabase_url}/storage/v1/object/"
        f"{bucket}/model-releases/{release_id}/{file_name}"
    )


def _storage_headers(
    settings: AppSettings, *, content_type: str | None = None
) -> dict[str, str]:
    headers = {
        "apikey": settings.supabase_service_role_key,
        "Authorization": f"Bearer {settings.supabase_service_role_key}",
    }
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def publish_transaction_release(
    settings: AppSettings,
    artifact_root: Path,
    *,
    release_id: str,
) -> ArtifactManifest:
    if not settings.supabase_configured or not settings.artifact_storage_bucket:
        raise ValueError(
            "Supabase URL, service role key, and artifact bucket are required"
        )
    manifest = build_transaction_release_manifest(artifact_root, release_id=release_id)
    manifest_path = write_manifest(artifact_root, manifest)
    upload_names = [MANIFEST_FILENAME, *REQUIRED_TRANSACTION_FILES]
    for name in upload_names:
        path = manifest_path if name == MANIFEST_FILENAME else artifact_root / name
        request = urllib.request.Request(
            _storage_url(settings, release_id, name),
            data=path.read_bytes(),
            method="POST",
            headers={
                **_storage_headers(settings, content_type="application/octet-stream"),
                "x-upsert": "false",
            },
        )
        try:
            with urllib.request.urlopen(
                request, timeout=settings.artifact_download_timeout_seconds
            ) as response:
                if response.status not in {200, 201}:
                    raise RuntimeError(f"Artifact upload failed for {name}")
        except urllib.error.HTTPError as error:
            if error.code == 409:
                raise FileExistsError(
                    f"Artifact release already exists: {release_id}"
                ) from error
            raise RuntimeError(
                f"Artifact upload failed for {name}: {error.code}"
            ) from error
    return manifest


def download_transaction_release(
    settings: AppSettings, release_id: str, destination: Path
) -> None:
    if not settings.supabase_configured or not settings.artifact_storage_bucket:
        raise ValueError("Supabase artifact storage is not configured")
    destination.mkdir(parents=True, exist_ok=True)
    for name in [MANIFEST_FILENAME, *REQUIRED_TRANSACTION_FILES]:
        request = urllib.request.Request(
            _storage_url(settings, release_id, name),
            method="GET",
            headers=_storage_headers(settings),
        )
        last_error: Exception | None = None
        for _attempt in range(max(settings.artifact_download_retries, 0) + 1):
            try:
                with urllib.request.urlopen(
                    request, timeout=settings.artifact_download_timeout_seconds
                ) as response:
                    (destination / name).write_bytes(response.read())
                    last_error = None
                    break
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
        if last_error is not None:
            raise RuntimeError(f"Artifact download failed for {name}") from last_error


def ensure_transaction_release(
    settings: AppSettings,
) -> tuple[Path, ArtifactManifest | None]:
    release_id = (
        settings.rollback_release_id or settings.transaction_artifact_release_id
    )
    if not release_id:
        root = settings.transaction_candidate_artifact_root
        manifest_path = root / MANIFEST_FILENAME
        if manifest_path.exists():
            return root, validate_release_directory(root, validate_model=False)
        return root, None

    active_root = settings.artifact_cache_root / release_id / "active"
    try:
        return active_root, validate_release_directory(
            active_root, expected_release_id=release_id, validate_model=False
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError):
        pass

    with tempfile.TemporaryDirectory(prefix=f"fraudguard-{release_id}-") as temp_name:
        staged = Path(temp_name) / "release"
        download_transaction_release(settings, release_id, staged)
        manifest = validate_release_directory(
            staged, expected_release_id=release_id, validate_model=False
        )
        target_parent = active_root.parent
        target_parent.mkdir(parents=True, exist_ok=True)
        next_root = target_parent / "next"
        if next_root.exists():
            shutil.rmtree(next_root)
        shutil.copytree(staged, next_root)
        if active_root.exists():
            backup = target_parent / "previous"
            if backup.exists():
                shutil.rmtree(backup)
            active_root.replace(backup)
        next_root.replace(active_root)
    return active_root, manifest
