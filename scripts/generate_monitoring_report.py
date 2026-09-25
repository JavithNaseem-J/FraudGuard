from __future__ import annotations

import argparse
from pathlib import Path

from FraudGuard.monitoring.evidently_reports import generate_output_monitoring_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a sanitized transaction-output drift report."
    )
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--current", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/monitoring"))
    parser.add_argument("--release-id", default="local")
    parser.add_argument("--min-rows", type=int, default=25)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metrics = generate_output_monitoring_report(
        args.reference,
        args.current,
        args.output_dir / "output_drift.html",
        metrics_path=args.output_dir / "output_drift_metrics.json",
        release_id=args.release_id,
        min_rows=args.min_rows,
    )
    print(f"Monitoring status: {metrics['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
