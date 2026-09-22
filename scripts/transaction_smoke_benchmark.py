from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from FraudGuard.data.transaction_benchmark import (  # noqa: E402
    default_transaction_data_config,
    run_transaction_smoke_benchmark,
)


def main() -> int:
    report = run_transaction_smoke_benchmark(
        default_transaction_data_config(sample_rows=75000)
    )
    print(
        f"Wrote transaction smoke benchmark report: {report['benchmark_report_path']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
