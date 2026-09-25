from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from FraudGuard.cloud.artifacts import (
    build_transaction_release_manifest,
    write_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    features = ["TransactionDT", "TransactionAmt"]
    frame = pd.DataFrame(
        {
            "TransactionDT": list(range(20)),
            "TransactionAmt": [float(index * 10 + 1) for index in range(20)],
            "isFraud": [0, 0, 0, 1] * 5,
        }
    )
    model = Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(
                    [
                        (
                            "numeric",
                            Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                            features,
                        )
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=200, random_state=42)),
        ]
    )
    model.fit(frame[features], frame["isFraud"])
    joblib.dump(model, args.output / "model.joblib")
    threshold = {
        "threshold": 0.4,
        "objective": "validation_cost_weighted_loss",
        "false_positive_cost": 1.0,
        "false_negative_cost": 20.0,
    }
    metadata = {
        "artifact_schema_version": 1,
        "model_version": "ci-fixture-v1",
        "created_at_utc": "2026-09-24T00:00:00+00:00",
        "model_name": "CI transaction fixture",
        "feature_names": features,
        "numeric_features": features,
        "categorical_features": [],
        "numeric_feature_count": len(features),
        "categorical_feature_count": 0,
        "public_test_used_for_metrics": False,
        "score_is_calibrated": False,
        "threshold": threshold,
        "promotion_gates": {"all_gates_passed": True},
    }
    audit = {
        "selected_feature_count": len(features),
        "public_test_used_for_metrics": False,
        "schema_risks": ["Synthetic CI fixture; not production model evidence."],
    }
    (args.output / "threshold.json").write_text(json.dumps(threshold), encoding="utf-8")
    (args.output / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (args.output / "feature_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    manifest = build_transaction_release_manifest(
        args.output, release_id="ci-fixture-v1", model_version="ci-fixture-v1"
    )
    write_manifest(args.output, manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
