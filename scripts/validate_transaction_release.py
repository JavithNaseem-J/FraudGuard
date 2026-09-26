from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from FraudGuard.pipeline.transaction_pipeline import TransactionPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a promoted transaction-only model against real public-test rows."
        )
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--public-test",
        type=Path,
        default=Path("data/test_transaction.csv"),
    )
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--expected-features", type=int, default=392)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.rows < 1000:
        raise SystemExit("Release validation requires at least 1,000 real rows")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    public_test = args.public_test
    if not public_test.exists():
        for candidate in [
            Path("/content/data/test_transaction.csv"),
            Path("/content/test_transaction.csv"),
            Path.cwd() / "data" / "test_transaction.csv",
            Path(__file__).resolve().parents[1] / "data" / "test_transaction.csv",
        ]:
            if candidate.exists():
                public_test = candidate
                break
    if not public_test.exists():
        raise SystemExit(f"Public transaction test file not found: {args.public_test}")
    args.public_test = public_test

    pipeline = TransactionPipeline(args.artifact_root)
    identity_features = [
        feature
        for feature in pipeline.feature_names
        if feature.startswith("id_") or feature in {"DeviceType", "DeviceInfo"}
    ]
    if len(pipeline.feature_names) != args.expected_features:
        raise SystemExit(
            "Unexpected model feature count: "
            f"expected {args.expected_features}, observed {len(pipeline.feature_names)}"
        )
    if identity_features:
        raise SystemExit(f"Identity features are prohibited: {identity_features}")

    frame = pd.read_csv(args.public_test, nrows=args.rows)
    if len(frame) < args.rows:
        raise SystemExit(
            f"Public test file has only {len(frame)} rows; {args.rows} are required"
        )
    missing = sorted(set(pipeline.feature_names) - set(frame.columns))
    if missing:
        raise SystemExit(f"Public test schema is missing model features: {missing}")

    scored_rows = 0
    selected = frame.loc[:, pipeline.feature_names]
    for start in range(0, len(selected), args.batch_size):
        rows = selected.iloc[start : start + args.batch_size].to_dict(orient="records")
        result = pipeline.predict_rows(rows)
        if len(result["results"]) != len(rows):
            raise SystemExit("Prediction result count does not match input row count")
        scored_rows += len(rows)

    print(
        json.dumps(
            {
                "status": "passed",
                "rows_scored": scored_rows,
                "feature_count": len(pipeline.feature_names),
                "identity_feature_count": 0,
                "public_test_used_for_metrics": False,
                "model_version": pipeline.model_version,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
