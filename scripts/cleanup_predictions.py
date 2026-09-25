from __future__ import annotations

import argparse
from datetime import UTC, datetime, timedelta

from FraudGuard.cloud.persistence import SupabasePersistence
from FraudGuard.cloud.settings import load_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Owner-only cleanup for sanitized demo prediction records."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--older-than-days",
        type=int,
        metavar="DAYS",
        help="Delete records older than this many days.",
    )
    group.add_argument(
        "--all",
        action="store_true",
        help="Delete every persisted prediction record.",
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Required acknowledgement for a destructive cleanup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.confirm:
        raise SystemExit("Refusing cleanup without --confirm")
    if args.older_than_days is not None and args.older_than_days < 1:
        raise SystemExit("--older-than-days must be at least 1")

    persistence = SupabasePersistence(load_settings())
    if not persistence.enabled:
        raise SystemExit("Supabase server credentials are required")

    if args.all:
        deleted = persistence.delete_all_predictions()
    else:
        cutoff = datetime.now(UTC) - timedelta(days=args.older_than_days)
        deleted = persistence.delete_predictions_before(cutoff.isoformat())
    if deleted is None:
        raise SystemExit("Cleanup failed")
    print(f"Deleted {deleted} sanitized prediction record(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
