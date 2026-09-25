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
        help="Bounded row count for workstation and CI evaluation.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use all labeled training rows instead of the bounded sample.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sample_rows = None if args.full else args.sample_rows
    report = run_transaction_strong_benchmark(
        default_transaction_data_config(sample_rows=sample_rows)
    )
    promotion = report["strong_benchmark"]["promotion_gates"]
    print(
        json.dumps(
            {
                "report": report["strong_benchmark_report_path"],
                "rows_mode": "full" if sample_rows is None else sample_rows,
                "promotion_decision": promotion["decision"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
