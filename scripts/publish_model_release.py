from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from FraudGuard.cloud.artifacts import (  # noqa: E402
    build_transaction_release_manifest,
    publish_transaction_release,
    validate_release_directory,
    write_manifest,
)
from FraudGuard.cloud.settings import load_settings  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and publish an immutable transaction model release."
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    parser.add_argument(
        "--local-only",
        action="store_true",
        help="Only write and validate the manifest; do not upload to Supabase Storage.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    settings = load_settings()
    manifest = build_transaction_release_manifest(
        args.artifact_root,
        release_id=args.release_id,
    )
    write_manifest(args.artifact_root, manifest)
    validate_release_directory(args.artifact_root, expected_release_id=args.release_id)
    if not args.local_only:
        manifest = publish_transaction_release(
            settings,
            args.artifact_root,
            release_id=args.release_id,
        )
    print(
        json.dumps(
            {
                "release_id": manifest.release_id,
                "model_version": manifest.model_version,
                "artifact_schema_version": manifest.artifact_schema_version,
                "file_count": len(manifest.files),
                "published": not args.local_only,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
