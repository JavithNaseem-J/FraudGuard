from __future__ import annotations

import argparse
import json

from FraudGuard.data.transaction_benchmark import (
    default_transaction_data_config,
    run_transaction_strong_benchmark,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run chronological transaction-model evaluation."
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=75_000,
        help="Bounded row count for a non-publishable workstation/CI diagnostic.",
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help=(
            "Run the publishable full-data workflow. Reads all 590,540 labeled "
            "rows and ignores --sample-rows."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_rows = None if args.release else args.sample_rows
    report = run_transaction_strong_benchmark(
        default_transaction_data_config(
            sample_rows=sample_rows,
            release_mode=args.release,
        )
    )
    promotion = report["strong_benchmark"]["promotion_gates"]
    print(
        json.dumps(
            {
                "report": report["strong_benchmark_report_path"],
                "execution_mode": "release" if args.release else "diagnostic",
                "rows": "all" if args.release else sample_rows,
                "promotion_decision": promotion["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
