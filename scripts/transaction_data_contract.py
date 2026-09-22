from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from FraudGuard.data.transaction_benchmark import (  # noqa: E402
    default_transaction_data_config,
    validate_transaction_data_contract,
)


def main() -> int:
    config = default_transaction_data_config()
    report = validate_transaction_data_contract(config)
    output_path = Path("artifacts/benchmark/transaction_data/contract_report.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["ready"]:
        print(json.dumps(report, indent=2))
        return 1
    print(f"Wrote transaction data contract report: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
