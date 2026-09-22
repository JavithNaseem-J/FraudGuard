import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app import app
from FraudGuard.cloud.artifacts import (
    build_transaction_release_manifest,
    validate_release_directory,
    write_manifest,
)
from FraudGuard.cloud.persistence import SupabasePersistence
from FraudGuard.cloud.rate_limit import CloudRateLimiter
from FraudGuard.cloud.settings import AppSettings
from FraudGuard.components.evaluation import Evaluation
from FraudGuard.components.preprocess import Transform
from FraudGuard.components.training import build_preprocessor, select_f1_threshold
from FraudGuard.data.dataset_registry import (
    default_dataset_registry,
    validate_dataset_paths,
)
from FraudGuard.data.dataset_registry import find_registered_dataset
from FraudGuard.data.transaction_benchmark import (
    TransactionBenchmarkConfig,
    load_labeled_transaction_data,
    prepare_transaction_benchmark,
    run_transaction_smoke_benchmark,
    run_transaction_strong_benchmark,
    split_labeled_transaction_data,
    validate_transaction_data_contract,
)
from FraudGuard.entity.config_entity import (
    DataTransformationConfig,
    ModelEvaluationConfig,
)
from FraudGuard.monitoring.evidently_reports import (
    generate_delayed_label_performance_report,
    generate_unlabeled_drift_report,
)
from FraudGuard.pipeline.inference_pipeline import PredictionPipeline
from FraudGuard.pipeline.transaction_candidate_pipeline import (
    TransactionCandidatePipeline,
)
from FraudGuard.utils.costs import select_cost_weighted_threshold
from FraudGuard.utils.helpers import init_mlflow_tracking, load_json, save_json


FEATURES = [
    "Transaction_Amount",
    "Time_of_Transaction",
    "Previous_Fraudulent_Transactions",
    "Account_Age",
    "Number_of_Transactions_Last_24H",
    "Transaction_Type",
    "Device_Used",
    "Location",
    "Payment_Method",
]
NUMERIC_FEATURES = FEATURES[:5]
CATEGORICAL_FEATURES = FEATURES[5:]


def make_transactions(rows: int = 40) -> pd.DataFrame:
    records = []
    for index in range(rows):
        records.append(
            {
                "Transaction_ID": f"txn-{index}",
                "User_ID": 1000 + index % 9,
                "Transaction_Amount": float(index + 1) if index != 3 else np.nan,
                "Time_of_Transaction": float(index % 24),
                "Previous_Fraudulent_Transactions": index % 4,
                "Account_Age": 10 + index,
                "Number_of_Transactions_Last_24H": 1 + index % 12,
                "Transaction_Type": ["purchase", "transfer"][index % 2],
                "Device_Used": ["mobile", "desktop"][index % 2],
                "Location": ["north", "south", None][index % 3],
                "Payment_Method": ["card", "wallet"][index % 2],
                "Fraudulent": 1 if index % 7 == 0 else 0,
            }
        )
    return pd.DataFrame(records)


def transformation_config(tmp_path: Path, data_path: Path) -> DataTransformationConfig:
    status_path = tmp_path / "validation.json"
    status_path.write_text('{"validation_status": true}', encoding="utf-8")
    return DataTransformationConfig(
        root_dir=tmp_path / "transform",
        data_path=data_path,
        validation_status_path=status_path,
        target_column="Fraudulent",
        columns_to_drop=["Transaction_ID", "User_ID"],
        test_size=0.25,
        random_state=42,
    )


def fitted_pipeline() -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    data = make_transactions(40).drop(columns=["Transaction_ID", "User_ID"])
    X = data[FEATURES]
    y = data["Fraudulent"]
    pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(CATEGORICAL_FEATURES, NUMERIC_FEATURES),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced", max_iter=500, random_state=42
                ),
            ),
        ]
    )
    pipeline.fit(X, y)
    return pipeline, X, y


def write_artifacts(root: Path) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    pipeline, X, y = fitted_pipeline()
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, root / "model.joblib")
    threshold = {
        "optimal_threshold": 0.4,
        "objective": "out_of_fold_f1",
        "precision": 0.5,
        "recall": 0.5,
        "f1": 0.5,
    }
    metadata = {
        "artifact_schema_version": 2,
        "version": "test",
        "model_name": "LogisticRegression",
        "feature_names": FEATURES,
        "categorical_columns": CATEGORICAL_FEATURES,
        "numeric_columns": NUMERIC_FEATURES,
        "target_column": "Fraudulent",
        "optimal_threshold": 0.4,
        "baseline_cv_score": float(y.mean()),
        "score_is_calibrated": False,
    }
    (root / "optimal_threshold.json").write_text(
        json.dumps(threshold), encoding="utf-8"
    )
    (root / "model_version.json").write_text(json.dumps(metadata), encoding="utf-8")
    return pipeline, X, y


def local_settings(
    tmp_path: Path,
    model_mode: str = "baseline",
    auth_required: bool = False,
    api_key: str = "",
    max_request_bytes: int = 1_048_576,
    max_batch_rows: int = 100,
) -> AppSettings:
    return AppSettings(
        app_env="test",
        model_mode=model_mode,
        model_artifact_root=tmp_path / "trainer",
        transaction_candidate_artifact_root=tmp_path / "candidate",
        transaction_artifact_release_id="",
        artifact_storage_bucket="",
        artifact_cache_root=tmp_path / "release-cache",
        artifact_download_timeout_seconds=5,
        artifact_download_retries=0,
        rollback_release_id="",
        public_base_url="http://testserver",
        port=8000,
        supabase_url="",
        supabase_service_role_key="",
        upstash_redis_rest_url="",
        upstash_redis_rest_token="",
        upstash_fail_closed=False,
        rate_limit_requests=2,
        rate_limit_window_seconds=60,
        prediction_persistence_enabled=True,
        auth_required=auth_required,
        fraudguard_api_key=api_key,
        max_request_bytes=max_request_bytes,
        max_batch_rows=max_batch_rows,
    )


def write_candidate_artifacts(root: Path) -> tuple[Pipeline, pd.DataFrame, pd.Series]:
    pipeline, X, y = fitted_pipeline()
    root.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, root / "model.joblib")
    threshold = {
        "threshold": 0.4,
        "objective": "out_of_fold_cost_weighted_loss",
        "false_positive_cost": 1.0,
        "false_negative_cost": 20.0,
    }
    metadata = {
        "artifact_schema_version": 1,
        "created_at_utc": "2026-09-21T00:00:00+00:00",
        "model_name": "LightGBM tabular transaction benchmark",
        "dataset": "transaction-data-local",
        "metrics_source": "internal labeled test split",
        "public_test_used_for_metrics": False,
        "serving_promotion": False,
        "feature_names": FEATURES,
        "numeric_feature_count": len(NUMERIC_FEATURES),
        "categorical_feature_count": len(CATEGORICAL_FEATURES),
        "threshold": threshold,
        "promotion_gates": {"all_gates_passed": True},
    }
    feature_audit = {
        "selected_feature_count": len(FEATURES),
        "numeric_feature_count": len(NUMERIC_FEATURES),
        "categorical_feature_count": len(CATEGORICAL_FEATURES),
        "public_test_used_for_metrics": False,
        "schema_risks": ["test schema risk"],
    }
    (root / "threshold.json").write_text(json.dumps(threshold), encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "feature_audit.json").write_text(
        json.dumps(feature_audit), encoding="utf-8"
    )
    return pipeline, X, y


def make_ieee_cis_files(tmp_path: Path) -> TransactionBenchmarkConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    transaction_rows = []
    for index in range(20):
        transaction_rows.append(
            {
                "TransactionID": 1000 + index,
                "isFraud": 1 if index % 4 == 0 else 0,
                "TransactionDT": index * 60,
                "TransactionAmt": float(10 + index),
                "card1": 100 + index % 3,
                "ProductCD": ["W", "C", "R"][index % 3],
                "V1": float(index % 5),
            }
        )
    pd.DataFrame(transaction_rows).to_csv(
        data_dir / "train_transaction.csv", index=False
    )
    pd.DataFrame(
        [
            {"TransactionID": 1000, "id_01": 1.0},
            {"TransactionID": 1001, "id_01": 2.0},
            {"TransactionID": 1004, "id_01": 3.0},
        ]
    ).to_csv(data_dir / "train_identity.csv", index=False)
    pd.DataFrame([{"TransactionID": 9000, "id_01": 5.0}]).to_csv(
        data_dir / "test_identity.csv", index=False
    )
    return TransactionBenchmarkConfig(
        train_transaction_path=data_dir / "train_transaction.csv",
        train_identity_path=data_dir / "train_identity.csv",
        public_test_identity_path=data_dir / "test_identity.csv",
        output_dir=tmp_path / "artifacts" / "benchmark" / "ieee_cis",
        test_size=0.25,
        validation_size=0.25,
        random_state=7,
    )


def test_split_is_deduplicated_deterministic_and_stratified(tmp_path):
    source = make_transactions(40)
    source = pd.concat([source, source.iloc[[0]]], ignore_index=True)
    data_path = tmp_path / "transactions.csv"
    source.to_csv(data_path, index=False)
    config = transformation_config(tmp_path, data_path)

    first_train, first_test = Transform(config).train_test_splitting()
    second_train, second_test = Transform(config).train_test_splitting()

    pd.testing.assert_frame_equal(first_train, second_train)
    pd.testing.assert_frame_equal(first_test, second_test)
    assert len(first_train) + len(first_test) == 40
    assert not first_train.duplicated().any()
    assert not first_test.duplicated().any()
    assert first_train.merge(first_test, how="inner").empty
    assert abs(first_train.Fraudulent.mean() - first_test.Fraudulent.mean()) < 0.08


def test_split_refuses_failed_validation(tmp_path):
    data_path = tmp_path / "transactions.csv"
    make_transactions().to_csv(data_path, index=False)
    config = transformation_config(tmp_path, data_path)
    config.validation_status_path.write_text(
        '{"validation_status": false}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="validation failed"):
        Transform(config).train_test_splitting()


def test_preprocessor_handles_missing_and_unknown_categories():
    train = make_transactions(20).drop(columns=["Transaction_ID", "User_ID"])
    preprocessor = build_preprocessor(CATEGORICAL_FEATURES, NUMERIC_FEATURES)
    preprocessor.fit(train[FEATURES])

    unseen = train.iloc[[0]][FEATURES].copy()
    unseen.loc[:, "Location"] = "never-seen-location"
    transformed = preprocessor.transform(unseen)
    transformed_values = (
        transformed.toarray() if hasattr(transformed, "toarray") else transformed
    )

    assert transformed.shape[0] == 1
    assert np.isfinite(transformed_values).all()


def test_threshold_selection_uses_supplied_training_scores():
    result = select_f1_threshold(np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9]))

    assert result["objective"] == "out_of_fold_f1"
    assert 0 <= result["optimal_threshold"] <= 1
    assert result["f1"] == pytest.approx(1.0)


def test_cost_weighted_threshold_uses_configured_costs():
    result = select_cost_weighted_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        false_positive_cost=1.0,
        false_negative_cost=20.0,
    )

    assert result["objective"] == "out_of_fold_cost_weighted_loss"
    assert result["false_positive_cost"] == 1.0
    assert result["false_negative_cost"] == 20.0
    assert result["average_cost"] == pytest.approx(0.0)


def test_evaluation_applies_stored_threshold_and_reports_imbalanced_metrics(tmp_path):
    artifact_root = tmp_path / "trainer"
    pipeline, X, y = write_artifacts(artifact_root)
    test_path = tmp_path / "test.csv"
    X.assign(Fraudulent=y.to_numpy()).to_csv(test_path, index=False)
    evaluation_root = tmp_path / "evaluation"
    config = ModelEvaluationConfig(
        root_dir=evaluation_root,
        test_path=test_path,
        model_path=artifact_root / "model.joblib",
        threshold_path=artifact_root / "optimal_threshold.json",
        model_version_path=artifact_root / "model_version.json",
        metrics_path=evaluation_root / "metrics.json",
        target_column="Fraudulent",
        cm_path=evaluation_root / "cm.png",
        roc_path=evaluation_root / "roc.png",
        pr_path=evaluation_root / "pr.png",
    )

    metrics = Evaluation(config).evaluation()
    expected = (pipeline.predict_proba(X)[:, 1] >= 0.4).astype(int)

    assert metrics["threshold"] == 0.4
    assert metrics["positive_support"] == int(y.sum())
    assert metrics["confusion_matrix"] == (
        pd.crosstab(
            pd.Categorical(y, categories=[0, 1]),
            pd.Categorical(expected, categories=[0, 1]),
            dropna=False,
        )
        .to_numpy()
        .tolist()
    )
    assert "average_precision" in metrics
    assert "brier_score" in metrics
    assert metrics["cost_weighted"]["false_negative_cost"] == 20.0
    assert (evaluation_root / "pr.png").exists()


def test_inference_contract_handles_unknowns_and_rejects_missing_features(tmp_path):
    artifact_root = tmp_path / "trainer"
    _, X, _ = write_artifacts(artifact_root)
    predictor = PredictionPipeline(artifact_root=artifact_root)
    request = X.iloc[[0]].copy()
    request.loc[:, "Location"] = "unseen-location"

    result = predictor.predict(request)

    assert result["fraud_status"] in {"Yes", "No"}
    assert result["threshold_used"] == 0.4
    assert result["model_version"] == "test"
    assert result["score_is_calibrated"] is False

    with pytest.raises(ValueError, match="Missing required features"):
        predictor.predict(request.drop(columns=["Location"]))


def test_inference_rejects_mismatched_threshold_artifact(tmp_path):
    artifact_root = tmp_path / "trainer"
    write_artifacts(artifact_root)
    (artifact_root / "optimal_threshold.json").write_text(
        '{"optimal_threshold": 0.9}', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="does not match"):
        PredictionPipeline(artifact_root=artifact_root)


def test_transaction_candidate_pipeline_validates_schema_and_orders_features(tmp_path):
    _, X, _ = write_candidate_artifacts(tmp_path / "candidate")
    predictor = TransactionCandidatePipeline(tmp_path / "candidate")
    row = X.iloc[0].to_dict()
    reversed_row = {key: row[key] for key in reversed(FEATURES)}
    reversed_row["unexpected_feature"] = "ignored"

    result = predictor.predict_rows([reversed_row])

    assert result["model_version"] == "2026-09-21T00:00:00+00:00"
    assert result["feature_count"] == len(FEATURES)
    assert result["ignored_features"] == ["unexpected_feature"]
    assert result["results"][0]["fraud_status"] in {"Yes", "No"}

    with pytest.raises(ValueError, match="Row 0 missing required features"):
        predictor.predict_rows([{key: row[key] for key in FEATURES[:-1]}])


def test_transaction_release_manifest_validates_checksums_and_paths(tmp_path):
    candidate_root = tmp_path / "candidate"
    write_candidate_artifacts(candidate_root)
    manifest = build_transaction_release_manifest(
        candidate_root,
        release_id="release-20260922",
    )
    write_manifest(candidate_root, manifest)

    loaded = validate_release_directory(
        candidate_root,
        expected_release_id="release-20260922",
    )

    assert loaded.release_id == "release-20260922"
    assert {item.path for item in loaded.files} == {
        "model.joblib",
        "threshold.json",
        "metadata.json",
        "feature_audit.json",
    }

    threshold_path = candidate_root / "threshold.json"
    threshold_bytes = bytearray(threshold_path.read_bytes())
    threshold_bytes[-2] = ord("9") if threshold_bytes[-2] != ord("9") else ord("8")
    threshold_path.write_bytes(threshold_bytes)
    with pytest.raises(ValueError, match="checksum mismatch"):
        validate_release_directory(
            candidate_root,
            expected_release_id="release-20260922",
            validate_model=False,
        )


def test_transaction_release_manifest_rejects_path_traversal(tmp_path):
    candidate_root = tmp_path / "candidate"
    write_candidate_artifacts(candidate_root)
    manifest = build_transaction_release_manifest(
        candidate_root,
        release_id="release-20260922",
    )
    payload = manifest.to_dict()
    payload["files"][0]["path"] = "../model.joblib"
    (candidate_root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe artifact path"):
        validate_release_directory(candidate_root, validate_model=False)


def test_json_helpers_and_disabled_mlflow_are_safe(tmp_path):
    test_file = tmp_path / "test.json"
    save_json(test_file, {"key": "value"})

    assert load_json(test_file) == {"key": "value"}
    init_mlflow_tracking()
    init_mlflow_tracking()


def test_cloud_settings_fallbacks_dataset_registry_and_rate_limit(tmp_path):
    settings = local_settings(tmp_path)
    persistence = SupabasePersistence(settings)
    limiter = CloudRateLimiter(settings)
    registry = default_dataset_registry(project_root=tmp_path)
    transaction_status = validate_dataset_paths(registry["transaction-data-local"])

    assert persistence.mode == "local_noop"
    assert limiter.check("client").allowed is True
    assert limiter.check("client").allowed is True
    assert limiter.check("client").allowed is False
    assert transaction_status["ready"] is False
    assert "train_transaction.csv" in transaction_status["missing_paths"][0]
    with pytest.raises(ValueError, match="retired from executable workflows"):
        find_registered_dataset(registry, "baseline-current")


def test_prediction_api_returns_safe_json_contract(tmp_path):
    artifact_root = tmp_path / "trainer"
    _, X, _ = write_artifacts(artifact_root)
    request = X.iloc[0].to_dict()
    payload = {
        "transaction_type": request["Transaction_Type"],
        "device_used": request["Device_Used"],
        "location": request["Location"],
        "payment_method": request["Payment_Method"],
        "transaction_amount": request["Transaction_Amount"],
        "time_of_transaction": request["Time_of_Transaction"],
        "previous_fraudulent_transactions": int(
            request["Previous_Fraudulent_Transactions"]
        ),
        "account_age": int(request["Account_Age"]),
        "number_of_transactions_last_24h": int(
            request["Number_of_Transactions_Last_24H"]
        ),
    }

    baseline_settings = local_settings(tmp_path, model_mode="baseline")
    with TestClient(app) as client:
        app.state.settings = local_settings(tmp_path, model_mode="baseline")
        app.state.predictor = PredictionPipeline(artifact_root=artifact_root)
        app.state.persistence = SupabasePersistence(baseline_settings)
        app.state.rate_limiter = CloudRateLimiter(baseline_settings)
        response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert "prediction_id" in body
    assert body["fraud_status"] in {"Yes", "No"}
    assert "data=" not in body["results_url"]
    assert body["persistence"] == "local_noop"


def test_prediction_api_requires_valid_api_key_when_enabled(tmp_path):
    artifact_root = tmp_path / "trainer"
    _, X, _ = write_artifacts(artifact_root)
    request = X.iloc[0].to_dict()
    payload = {
        "transaction_type": request["Transaction_Type"],
        "device_used": request["Device_Used"],
        "location": request["Location"],
        "payment_method": request["Payment_Method"],
        "transaction_amount": request["Transaction_Amount"],
        "time_of_transaction": request["Time_of_Transaction"],
        "previous_fraudulent_transactions": int(
            request["Previous_Fraudulent_Transactions"]
        ),
        "account_age": int(request["Account_Age"]),
        "number_of_transactions_last_24h": int(
            request["Number_of_Transactions_Last_24H"]
        ),
    }
    secure_settings = local_settings(
        tmp_path, auth_required=True, api_key="test-secret"
    )

    with TestClient(app) as client:
        app.state.settings = secure_settings
        app.state.predictor = PredictionPipeline(artifact_root=artifact_root)
        app.state.persistence = SupabasePersistence(secure_settings)
        app.state.rate_limiter = CloudRateLimiter(secure_settings)

        assert client.get("/ready").status_code == 200
        missing = client.post("/predict", json=payload)
        wrong = client.post(
            "/predict", json=payload, headers={"x-api-key": "wrong-secret"}
        )
        valid = client.post(
            "/predict", json=payload, headers={"x-api-key": "test-secret"}
        )
        bearer = client.post(
            "/predict",
            json=payload,
            headers={"Authorization": "Bearer test-secret"},
        )

    assert missing.status_code == 401
    assert wrong.status_code == 403
    assert valid.status_code == 200
    assert bearer.status_code == 200


def test_request_size_guard_rejects_large_payload_before_prediction(tmp_path):
    settings = local_settings(tmp_path, max_request_bytes=20)
    payload = {
        "transaction_type": "purchase",
        "device_used": "mobile",
        "location": "north",
        "payment_method": "card",
        "transaction_amount": 10.0,
        "time_of_transaction": 1.0,
        "previous_fraudulent_transactions": 0,
        "account_age": 100,
        "number_of_transactions_last_24h": 1,
    }

    with TestClient(app) as client:
        app.state.settings = settings
        app.state.persistence = SupabasePersistence(settings)
        response = client.post("/predict", json=payload)

    assert response.status_code == 413
    assert response.json()["max_request_bytes"] == 20


def test_transaction_candidate_endpoint_is_feature_flagged_and_batch_safe(tmp_path):
    artifact_root = tmp_path / "trainer"
    _, X, _ = write_artifacts(artifact_root)
    write_candidate_artifacts(tmp_path / "candidate")
    row = X.iloc[0].to_dict()

    with TestClient(app) as client:
        app.state.settings = local_settings(tmp_path)
        app.state.predictor = PredictionPipeline(artifact_root=artifact_root)
        app.state.transaction_candidate = None
        app.state.persistence = SupabasePersistence(local_settings(tmp_path))
        app.state.rate_limiter = CloudRateLimiter(local_settings(tmp_path))

        disabled_response = client.post("/predict/transactions", json={"rows": [row]})
        assert disabled_response.status_code == 503

        candidate_settings = local_settings(
            tmp_path, model_mode="transaction_candidate"
        )
        app.state.settings = candidate_settings
        candidate_not_ready = client.get("/ready")
        assert candidate_not_ready.status_code == 503
        assert candidate_not_ready.json()["detail"]["candidate_model_loaded"] is False

        app.state.transaction_candidate = TransactionCandidatePipeline(
            candidate_settings.transaction_candidate_artifact_root
        )
        app.state.rate_limiter = CloudRateLimiter(candidate_settings)

        response = client.post(
            "/predict/transactions",
            json={"rows": [{**row, "unexpected_feature": "ignored"}]},
        )
        schema_response = client.get("/schema/transactions")

    assert response.status_code == 200
    body = response.json()
    assert body["model_mode"] == "transaction_candidate"
    assert body["row_count"] == 1
    assert body["ignored_features"] == ["unexpected_feature"]
    assert body["results"][0]["fraud_status"] in {"Yes", "No"}
    assert "prediction_id" in body["results"][0]
    assert schema_response.status_code == 200
    schema = schema_response.json()
    assert schema["model_mode"] == "transaction_candidate"
    assert schema["feature_names"] == list(X.columns)
    assert schema["feature_count"] == len(X.columns)


def test_home_page_targets_transaction_batch_console():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert "Transaction risk console" in response.text
    assert "/predict/transactions" in response.text
    assert 'action="/predict"' not in response.text


def test_version_endpoint_reports_non_sensitive_build_metadata(monkeypatch):
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.delenv("APP_COMMIT_SHA", raising=False)
    monkeypatch.delenv("APP_BUILD_TIME", raising=False)
    monkeypatch.setenv("BUILD_COMMIT_SHA", "a" * 40)
    monkeypatch.setenv("BUILD_TIME", "2026-09-22T00:00:00Z")

    with TestClient(app) as client:
        response = client.get("/version")

    assert response.status_code == 200
    assert response.json() == {
        "commit_sha": "a" * 40,
        "build_time": "2026-09-22T00:00:00Z",
    }


def test_transaction_mode_readiness_does_not_require_baseline_artifacts(tmp_path):
    _, X, _ = write_artifacts(tmp_path / "trainer")
    write_candidate_artifacts(tmp_path / "candidate")
    candidate_settings = local_settings(tmp_path, model_mode="transaction_candidate")

    with TestClient(app) as client:
        app.state.settings = candidate_settings
        app.state.predictor = None
        app.state.transaction_candidate = TransactionCandidatePipeline(
            candidate_settings.transaction_candidate_artifact_root
        )
        app.state.persistence = SupabasePersistence(candidate_settings)
        app.state.rate_limiter = CloudRateLimiter(candidate_settings)

        ready = client.get("/ready")
        legacy_response = client.post(
            "/predict",
            json={
                "transaction_type": X.iloc[0]["Transaction_Type"],
                "device_used": X.iloc[0]["Device_Used"],
                "location": X.iloc[0]["Location"],
                "payment_method": X.iloc[0]["Payment_Method"],
                "transaction_amount": X.iloc[0]["Transaction_Amount"],
                "time_of_transaction": X.iloc[0]["Time_of_Transaction"],
                "previous_fraudulent_transactions": int(
                    X.iloc[0]["Previous_Fraudulent_Transactions"]
                ),
                "account_age": int(X.iloc[0]["Account_Age"]),
                "number_of_transactions_last_24h": int(
                    X.iloc[0]["Number_of_Transactions_Last_24H"]
                ),
            },
        )

    assert ready.status_code == 200
    body = ready.json()
    assert body["baseline_model_loaded"] is False
    assert body["candidate_model_loaded"] is True
    assert legacy_response.status_code == 503


def test_transaction_candidate_endpoint_enforces_batch_row_limit(tmp_path):
    artifact_root = tmp_path / "trainer"
    _, X, _ = write_artifacts(artifact_root)
    write_candidate_artifacts(tmp_path / "candidate")
    row = X.iloc[0].to_dict()
    candidate_settings = local_settings(
        tmp_path, model_mode="transaction_candidate", max_batch_rows=1
    )

    with TestClient(app) as client:
        app.state.settings = candidate_settings
        app.state.predictor = PredictionPipeline(artifact_root=artifact_root)
        app.state.transaction_candidate = TransactionCandidatePipeline(
            candidate_settings.transaction_candidate_artifact_root
        )
        app.state.persistence = SupabasePersistence(candidate_settings)
        app.state.rate_limiter = CloudRateLimiter(candidate_settings)
        response = client.post("/predict/transactions", json={"rows": [row, row]})

    assert response.status_code == 422
    assert "Batch row count exceeds limit" in response.json()["detail"]


def test_monitoring_reports_handle_insufficient_and_delayed_labels(tmp_path):
    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    drift_metrics = tmp_path / "drift_metrics.json"
    pd.DataFrame(
        [
            {
                "prediction_id": "p1",
                "model_version": "v1",
                "release_id": "r1",
                "score": 0.1,
                "threshold": 0.4,
                "decision": "No",
                "raw_payload": "must-not-pass",
            }
        ]
    ).to_csv(reference, index=False)
    pd.DataFrame(
        [
            {
                "prediction_id": "p2",
                "model_version": "v1",
                "release_id": "r1",
                "score": 0.8,
                "threshold": 0.4,
                "decision": "Yes",
            }
        ]
    ).to_csv(current, index=False)

    generate_unlabeled_drift_report(
        reference,
        current,
        tmp_path / "drift.html",
        metrics_path=drift_metrics,
        min_rows=5,
    )
    drift_payload = json.loads(drift_metrics.read_text(encoding="utf-8"))

    assert drift_payload["status"] == "insufficient_data"
    assert drift_payload["reference_rows"] == 1

    predictions = tmp_path / "predictions.csv"
    feedback = tmp_path / "feedback.csv"
    performance_metrics = tmp_path / "performance.json"
    pd.DataFrame(
        [
            {"prediction_id": "p1", "score": 0.1, "threshold": 0.4, "decision": "No"},
            {"prediction_id": "p2", "score": 0.8, "threshold": 0.4, "decision": "Yes"},
            {"prediction_id": "p3", "score": 0.7, "threshold": 0.4, "decision": "Yes"},
        ]
    ).to_csv(predictions, index=False)
    pd.DataFrame(
        [
            {"prediction_id": "p1", "confirmed_label": 0},
            {"prediction_id": "p2", "confirmed_label": 1},
        ]
    ).to_csv(feedback, index=False)

    metrics = generate_delayed_label_performance_report(
        predictions,
        feedback,
        performance_metrics,
    )

    assert metrics["status"] == "success"
    assert metrics["label_count"] == 2
    assert metrics["label_coverage"] == pytest.approx(2 / 3)
    assert metrics["precision"] == pytest.approx(1.0)


def test_ieee_cis_contract_and_identity_left_join(tmp_path):
    config = make_ieee_cis_files(tmp_path)
    status = validate_transaction_data_contract(config)
    prepared = load_labeled_transaction_data(config)

    assert status["ready"] is True
    assert status["public_test_used_for_metrics"] is False
    assert status["class_counts"]["0"] == 15
    assert status["class_counts"]["1"] == 5
    assert len(prepared) == 20
    assert prepared["id_01"].isna().sum() == 17


def test_ieee_cis_splits_are_deterministic_and_internal(tmp_path):
    config = make_ieee_cis_files(tmp_path)
    prepared = load_labeled_transaction_data(config)
    first = split_labeled_transaction_data(prepared, config)
    second = split_labeled_transaction_data(prepared, config)
    report = prepare_transaction_benchmark(config)

    for split_name in ["train", "validation", "test"]:
        pd.testing.assert_frame_equal(first[split_name], second[split_name])
        assert first[split_name]["isFraud"].nunique() == 2

    assert report["public_test_used_for_metrics"] is False
    assert set(report["output_paths"]) == {"train", "validation", "test"}
    assert Path(report["report_path"]).exists()


def test_ieee_cis_smoke_benchmark_reports_cost_weighted_metrics(tmp_path):
    config = make_ieee_cis_files(tmp_path)
    report = run_transaction_smoke_benchmark(config)
    metrics = report["smoke_benchmark"]["metrics"]

    assert report["smoke_benchmark"]["metrics_source"] == "internal labeled test split"
    assert report["smoke_benchmark"]["serving_promotion"] is False
    assert metrics["rows"] == 5
    assert metrics["cost_weighted"]["false_negative_cost"] == 20.0
    assert "average_precision" in metrics
    assert Path(report["benchmark_report_path"]).exists()


def test_transaction_strong_benchmark_compares_against_smoke_baseline(tmp_path):
    pytest.importorskip("lightgbm")
    config = make_ieee_cis_files(tmp_path)
    serving_artifact = tmp_path / "artifacts" / "trainer" / "model.joblib"
    serving_artifact.parent.mkdir(parents=True, exist_ok=True)
    serving_artifact.write_bytes(b"current-serving-model")

    report = run_transaction_strong_benchmark(config)
    candidate_artifacts = report["strong_benchmark"]["candidate_artifacts"]

    assert report["strong_benchmark"]["serving_promotion"] is False
    assert report["model_comparison"]["promotion_decision"] == "not_promoted"
    assert "average_precision_delta" in report["model_comparison"]
    assert report["strong_benchmark"]["metrics"]["feature_count"] > 0
    assert report["strong_benchmark"]["promotion_gates"]["serving_promotion"] is False
    assert (
        report["strong_benchmark"]["feature_audit"]["public_test_used_for_metrics"]
        is False
    )
    assert Path(candidate_artifacts["model"]).exists()
    assert Path(candidate_artifacts["threshold"]).exists()
    assert Path(candidate_artifacts["metadata"]).exists()
    assert Path(candidate_artifacts["feature_audit"]).exists()
    assert Path(report["strong_benchmark_report_path"]).exists()
    assert serving_artifact.read_bytes() == b"current-serving-model"
