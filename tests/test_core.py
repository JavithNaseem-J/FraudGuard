import asyncio
import json
from dataclasses import replace
from pathlib import Path

import app as app_module
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from app import _dashboard_snapshot, app
from FraudGuard.cloud.artifacts import (
    build_transaction_release_manifest,
    publish_transaction_release,
    validate_release_directory,
    write_manifest,
)
from FraudGuard.cloud.persistence import PredictionRecord, SupabasePersistence
from FraudGuard.cloud.rate_limit import CloudRateLimiter
from FraudGuard.cloud.settings import AppSettings, load_settings
from FraudGuard.cloud.supabase import supabase_api_headers
from FraudGuard.data.transaction_benchmark import (
    COST_SENSITIVITY_RATIOS,
    LIGHTGBM_SEARCH_SPACE,
    TransactionBenchmarkConfig,
    prepare_transaction_benchmark,
    run_transaction_smoke_benchmark,
    run_transaction_strong_benchmark,
    split_labeled_transaction_data,
    validate_transaction_data_contract,
)
from FraudGuard.monitoring.evidently_reports import (
    APPROVED_MONITORING_COLUMNS,
    generate_output_monitoring_report,
    sanitize_monitoring_frame,
)
from FraudGuard.pipeline.transaction_pipeline import TransactionPipeline
from FraudGuard.utils.costs import select_cost_weighted_threshold

FEATURES = ["TransactionDT", "TransactionAmt", "card1"]


def local_settings(tmp_path: Path, **changes) -> AppSettings:
    settings = AppSettings(
        app_env="test",
        transaction_artifact_root=tmp_path / "model",
        transaction_artifact_release_id="",
        artifact_storage_bucket="",
        artifact_cache_root=tmp_path / "release-cache",
        artifact_download_timeout_seconds=5,
        artifact_download_retries=0,
        port=8000,
        supabase_url="",
        supabase_service_role_key="",
        upstash_redis_rest_url="",
        upstash_redis_rest_token="",
        upstash_fail_closed=False,
        rate_limit_requests=5,
        rate_limit_window_seconds=60,
        max_request_bytes=1_048_576,
        max_batch_rows=100,
        dashboard_retention_days=30,
        dashboard_row_limit=10_000,
    )
    return replace(settings, **changes)


def write_model_artifacts(root: Path, *, approved: bool = True) -> pd.DataFrame:
    root.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(
        {
            "TransactionDT": np.arange(40),
            "TransactionAmt": np.arange(40, dtype=float) + 1,
            "card1": np.arange(40) % 4,
            "isFraud": [0, 0, 0, 1] * 10,
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
                            FEATURES,
                        )
                    ]
                ),
            ),
            ("classifier", LogisticRegression(max_iter=300, random_state=42)),
        ]
    )
    model.fit(frame[FEATURES], frame["isFraud"])
    joblib.dump(model, root / "model.joblib")
    threshold = {
        "threshold": 0.4,
        "objective": "validation_cost_weighted_loss",
        "false_positive_cost": 1.0,
        "false_negative_cost": 20.0,
    }
    metadata = {
        "artifact_schema_version": 1,
        "model_version": "test-model-v1",
        "created_at_utc": "2026-09-24T00:00:00+00:00",
        "model_name": "Test transaction model",
        "feature_names": FEATURES,
        "numeric_features": FEATURES,
        "categorical_features": [],
        "numeric_feature_count": len(FEATURES),
        "categorical_feature_count": 0,
        "public_test_used_for_metrics": False,
        "score_is_calibrated": False,
        "threshold": threshold,
        "promotion_gates": {"all_gates_passed": approved},
    }
    audit = {
        "selected_feature_count": len(FEATURES),
        "public_test_used_for_metrics": False,
        "schema_risks": [],
    }
    (root / "threshold.json").write_text(json.dumps(threshold), encoding="utf-8")
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    (root / "feature_audit.json").write_text(json.dumps(audit), encoding="utf-8")
    return frame


def make_transaction_config(
    tmp_path: Path,
    rows: int = 120,
    *,
    release_mode: bool = False,
    with_public_test: bool = False,
) -> TransactionBenchmarkConfig:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    records = []
    for index in range(rows):
        records.append(
            {
                "TransactionID": 10_000 + index,
                "TransactionDT": index // 2,
                "TransactionAmt": float(10 + index),
                "card1": 100 + index % 5,
                "ProductCD": ["W", "C", "R"][index % 3],
                "isFraud": 1 if index % 5 == 0 else 0,
            }
        )
    frame = pd.DataFrame(records)
    frame.to_csv(data_dir / "train_transaction.csv", index=False)
    public_test_path = data_dir / "test_transaction.csv"
    if with_public_test:
        frame.drop(columns="isFraud").to_csv(public_test_path, index=False)
    return TransactionBenchmarkConfig(
        train_transaction_path=data_dir / "train_transaction.csv",
        public_test_transaction_path=(public_test_path if with_public_test else None),
        output_dir=tmp_path / "artifacts" / "benchmark" / "transaction_data",
        release_mode=release_mode,
        expected_full_source_rows=rows,
        expected_feature_count=4,
    )


class FakePersistence:
    mode = "supabase"

    def __init__(self, records=None):
        self.records = records or []
        self.persisted: list[PredictionRecord] = []

    def persist_predictions(self, records):
        self.persisted.extend(records)
        return len(records)

    def fetch_recent_predictions(self, *, cutoff, limit):
        return self.records[:limit]

    def delete_predictions_before(self, cutoff):
        return 0


def configure_test_app(tmp_path: Path, **setting_changes):
    frame = write_model_artifacts(tmp_path / "model")
    settings = local_settings(tmp_path, **setting_changes)
    app.state.settings = settings
    app.state.transaction_model = TransactionPipeline(
        settings.transaction_artifact_root
    )
    app.state.release_id = "test-release-v1"
    app.state.persistence = FakePersistence()
    app.state.rate_limiter = CloudRateLimiter(settings)
    app.state.readiness_error = None
    return frame


def test_cost_weighted_threshold_uses_configured_costs():
    result = select_cost_weighted_threshold(
        np.array([0, 0, 1, 1]),
        np.array([0.1, 0.4, 0.6, 0.9]),
        false_positive_cost=1.0,
        false_negative_cost=20.0,
    )
    assert result["false_negative_cost"] == 20.0
    assert result["average_cost"] == pytest.approx(0.0)


def test_version_endpoint_exposes_only_build_identity(monkeypatch, tmp_path):
    commit_sha = "a" * 40
    build_time = "2026-09-25T07:30:00Z"
    build_time_file = tmp_path / "build-time.txt"
    build_time_file.write_text(build_time, encoding="utf-8")
    monkeypatch.setattr(app_module, "BUILD_TIME_FILE", build_time_file)
    monkeypatch.delenv("RENDER_GIT_COMMIT", raising=False)
    monkeypatch.delenv("APP_COMMIT_SHA", raising=False)
    monkeypatch.setenv("BUILD_COMMIT_SHA", commit_sha)
    monkeypatch.delenv("APP_BUILD_TIME", raising=False)
    monkeypatch.setenv("BUILD_TIME", "unknown")

    assert asyncio.run(app_module.version()) == {
        "commit_sha": commit_sha,
        "build_time": build_time,
    }


def test_transaction_pipeline_validates_and_orders_features(tmp_path):
    frame = write_model_artifacts(tmp_path / "model")
    model = TransactionPipeline(tmp_path / "model")
    row = frame.iloc[0][FEATURES].to_dict()
    result = model.predict_rows([{**dict(reversed(list(row.items()))), "extra": 1}])
    assert result["model_version"] == "test-model-v1"
    assert result["ignored_features"] == ["extra"]
    assert result["results"][0]["validation_status"] == "valid"
    with pytest.raises(ValueError, match="missing required features"):
        model.predict_rows([{"TransactionDT": 1}])


def test_release_manifest_checks_integrity(tmp_path):
    root = tmp_path / "model"
    write_model_artifacts(root)
    manifest = build_transaction_release_manifest(root, release_id="release-v1")
    write_manifest(root, manifest)
    assert validate_release_directory(root).release_id == "release-v1"
    path = root / "threshold.json"
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="size mismatch|checksum mismatch"):
        validate_release_directory(root, validate_model=False)


def test_release_publication_uploads_manifest_last(monkeypatch, tmp_path):
    root = tmp_path / "model"
    write_model_artifacts(root)
    uploaded: list[str] = []

    class FakeResponse:
        status = 201

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def fake_urlopen(request, timeout):
        assert timeout == 5
        uploaded.append(request.full_url.rsplit("/", maxsplit=1)[-1])
        return FakeResponse()

    monkeypatch.setattr(
        "FraudGuard.cloud.artifacts.urllib.request.urlopen", fake_urlopen
    )
    settings = local_settings(
        tmp_path,
        supabase_url="https://example.supabase.co",
        supabase_service_role_key="sb_secret_example",
        artifact_storage_bucket="model-releases",
    )
    publish_transaction_release(settings, root, release_id="tx-test-release")
    assert uploaded[-1] == "manifest.json"
    assert set(uploaded[:-1]) == {
        "model.joblib",
        "threshold.json",
        "metadata.json",
        "feature_audit.json",
    }


def test_transaction_pipeline_rejects_unapproved_artifact(tmp_path):
    root = tmp_path / "model"
    write_model_artifacts(root, approved=False)
    with pytest.raises(ValueError, match="not approved"):
        TransactionPipeline(root)


def test_supabase_headers_support_modern_and_legacy_server_keys():
    modern = supabase_api_headers("sb_secret_example")
    assert modern == {"apikey": "sb_secret_example"}

    legacy = supabase_api_headers("eyJlegacy.jwt.value")
    assert legacy == {
        "apikey": "eyJlegacy.jwt.value",
        "Authorization": "Bearer eyJlegacy.jwt.value",
    }


def test_settings_have_bounded_demo_defaults(monkeypatch):
    for name in (
        "RATE_LIMIT_REQUESTS",
        "FRAUD_MODEL_MODE",
        "AUTH_REQUIRED",
        "FRAUDGUARD_API_KEY",
        "ROLLBACK_RELEASE_ID",
    ):
        monkeypatch.delenv(name, raising=False)
    settings = load_settings()
    assert settings.rate_limit_requests == 5
    assert settings.max_batch_rows == 100
    assert not hasattr(settings, "auth_required")
    assert not hasattr(settings, "model_mode")


def test_local_rate_limit_applies_before_prediction(tmp_path):
    settings = local_settings(tmp_path, rate_limit_requests=2)
    limiter = CloudRateLimiter(settings)
    assert limiter.check("client").allowed
    assert limiter.check("client").allowed
    assert not limiter.check("client").allowed


@pytest.mark.integration
def test_anonymous_prediction_persists_only_sanitized_fields(tmp_path):
    with TestClient(app) as client:
        frame = configure_test_app(tmp_path)
        persistence = app.state.persistence
        row = frame.iloc[0][FEATURES].to_dict()
        response = client.post("/predict/transactions", json={"rows": [row]})
    assert response.status_code == 200
    assert response.json()["release_id"] == "test-release-v1"
    assert len(persistence.persisted) == 1
    record = persistence.persisted[0]
    assert record.transaction_amount == row["TransactionAmt"]
    assert set(record.__dict__) == {
        "prediction_id",
        "request_id",
        "model_version",
        "release_id",
        "transaction_amount",
        "score",
        "threshold",
        "decision",
        "score_is_calibrated",
        "latency_ms",
        "created_at",
    }


@pytest.mark.integration
def test_prediction_guardrails_cover_rate_bytes_and_rows(tmp_path):
    with TestClient(app) as client:
        frame = configure_test_app(
            tmp_path,
            rate_limit_requests=1,
            max_request_bytes=200,
            max_batch_rows=1,
        )
        row = frame.iloc[0][FEATURES].to_dict()
        first = client.post("/predict/transactions", json={"rows": [row]})
        limited = client.post("/predict/transactions", json={"rows": [row]})
        app.state.rate_limiter = CloudRateLimiter(app.state.settings)
        too_many = client.post("/predict/transactions", json={"rows": [row, row]})
        oversized = client.post(
            "/predict/transactions",
            content=json.dumps({"rows": [{**row, "padding": "x" * 500}]}),
            headers={"content-type": "application/json"},
        )
    assert first.status_code == 200
    assert limited.status_code == 429
    assert too_many.status_code == 422
    assert oversized.status_code == 413


@pytest.mark.integration
def test_readiness_liveness_and_removed_enterprise_routes(tmp_path):
    with TestClient(app) as client:
        configure_test_app(tmp_path)
        assert client.get("/live").status_code == 200
        ready = client.get("/ready")
        assert ready.status_code == 200
        assert ready.json()["release_id"] == "test-release-v1"
        assert ready.json()["rate_limit_policy"]["requests"] == 5
        assert "schema_risks" not in ready.json()["model"]
        assert "schema_risks" not in client.get("/schema/transactions").json()
        assert client.post("/feedback", json={}).status_code == 404
        api = client.get("/api").json()
    assert api["access"] == "anonymous_rate_limited_demo"
    assert "feedback" not in api["endpoints"]
    assert "authentication" not in api


@pytest.mark.integration
def test_unavailable_model_fails_readiness_but_not_liveness(tmp_path):
    with TestClient(app) as client:
        app.state.settings = local_settings(tmp_path)
        app.state.transaction_model = None
        app.state.readiness_error = "FileNotFoundError"
        app.state.persistence = FakePersistence()
        app.state.rate_limiter = CloudRateLimiter(app.state.settings)
        assert client.get("/live").status_code == 200
        response = client.get("/ready")
    assert response.status_code == 503
    assert "FileNotFoundError" in response.text
    assert str(tmp_path) not in response.text


def test_dashboard_is_bounded_aggregated_and_redacted():
    records = [
        {
            "prediction_id": f"prediction-{index:02d}",
            "request_id": "request-secret-not-returned",
            "created_at": f"2026-09-{24 - index:02d}T10:00:00+00:00",
            "transaction_amount": 100 + index,
            "score": 0.9 if index < 6 else 0.1,
            "threshold": 0.4,
            "decision": "Yes" if index < 6 else "No",
            "latency_ms": 10,
            "model_version": "v1",
            "release_id": "r1",
        }
        for index in range(8)
    ]
    snapshot = _dashboard_snapshot(
        records,
        persistence_mode="supabase",
        window_start=pd.Timestamp("2026-08-25", tz="UTC").to_pydatetime(),
        window_end=pd.Timestamp("2026-09-24", tz="UTC").to_pydatetime(),
        truncated=True,
    )
    assert snapshot["transaction_count"] == 8
    assert snapshot["flagged_count"] == 6
    assert len(snapshot["recent_flagged"]) == 5
    assert snapshot["truncated"] is True
    assert "request-secret-not-returned" not in json.dumps(snapshot)
    assert all(len(row["prediction_id"]) <= 8 for row in snapshot["recent_flagged"])


@pytest.mark.integration
def test_dashboard_local_fallback_is_explicit(tmp_path):
    with TestClient(app) as client:
        configure_test_app(tmp_path)
        app.state.persistence = SupabasePersistence(app.state.settings)
        response = client.get("/dashboard")
    assert response.status_code == 200
    assert response.json()["persistence"] == "local_noop"
    assert response.json()["transaction_count"] == 0


def test_chronological_split_is_deterministic_and_keeps_time_groups(tmp_path):
    config = make_transaction_config(tmp_path)
    frame = pd.read_csv(config.train_transaction_path)
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    first = split_labeled_transaction_data(frame, config)
    second = split_labeled_transaction_data(frame, config)
    for name in ("train", "validation", "test"):
        pd.testing.assert_frame_equal(first[name], second[name])
        assert first[name]["isFraud"].nunique() == 2
    assert (
        first["train"]["TransactionDT"].max()
        < first["validation"]["TransactionDT"].min()
    )
    assert (
        first["validation"]["TransactionDT"].max()
        < first["test"]["TransactionDT"].min()
    )
    assert sum(map(len, first.values())) == len(frame.drop_duplicates())


def test_chronological_split_fails_if_a_period_has_one_class(tmp_path):
    config = make_transaction_config(tmp_path, rows=30)
    frame = pd.read_csv(config.train_transaction_path)
    frame["isFraud"] = 0
    frame.loc[:2, "isFraud"] = 1
    with pytest.raises(ValueError, match="partition must contain both"):
        split_labeled_transaction_data(frame, config)


def test_transaction_contract_and_smoke_evaluation_are_temporal(tmp_path):
    config = make_transaction_config(tmp_path)
    contract = validate_transaction_data_contract(config)
    prepared = prepare_transaction_benchmark(config)
    report = run_transaction_smoke_benchmark(config)
    assert contract["ready"] is True
    assert prepared["split_strategy"] == "chronological_70_15_15_time_groups"
    assert report["smoke_benchmark"]["threshold_selected_from"] == (
        "chronological validation period"
    )
    ratios = {
        item["false_negative_cost"]
        for item in report["smoke_benchmark"]["cost_sensitivity"]
    }
    assert ratios == set(COST_SENSITIVITY_RATIOS)
    assert report["smoke_benchmark"]["metrics_source"] == (
        "untouched chronological test period"
    )


def test_release_mode_requires_complete_source_and_public_schema(tmp_path):
    config = make_transaction_config(
        tmp_path,
        release_mode=True,
        with_public_test=True,
    )
    contract = validate_transaction_data_contract(config)
    prepared = prepare_transaction_benchmark(config)
    assert contract["ready"] is True
    assert contract["source_rows"] == 120
    assert len(contract["source_sha256"]) == 64
    assert contract["feature_count"] == 4
    assert contract["public_test_schema_compatible"] is True
    assert prepared["mode"] == "release"
    assert prepared["release_eligible_execution"] is True
    assert not any(config.output_dir.glob("train.csv"))

    truncated = replace(config, expected_full_source_rows=121)
    invalid = validate_transaction_data_contract(truncated)
    assert invalid["ready"] is False
    with pytest.raises(ValueError, match="contract failed"):
        prepare_transaction_benchmark(truncated)


def test_identity_feature_is_rejected_by_transaction_contract(tmp_path):
    config = make_transaction_config(tmp_path, with_public_test=True)
    frame = pd.read_csv(config.train_transaction_path)
    frame["id_01"] = 0
    frame.to_csv(config.train_transaction_path, index=False)
    contract = validate_transaction_data_contract(config)
    assert contract["ready"] is False
    assert contract["identity_features"] == ["id_01"]


def test_strong_benchmark_records_promotion_gates(tmp_path):
    pytest.importorskip("lightgbm")
    config = make_transaction_config(tmp_path)
    report = run_transaction_strong_benchmark(config)
    evidence = report["strong_benchmark"]
    assert evidence["promotion_gates"]["decision"] in {"approved", "blocked"}
    assert set(evidence["promotion_gates"]["gates"]) == {
        "average_precision",
        "recall",
        "average_cost",
        "logistic_baseline_cost",
        "feature_schema",
        "artifact_package",
        "full_data_mode",
        "final_holdout_isolation",
    }
    assert evidence["promotion_gates"]["gates"]["full_data_mode"]["passed"] is False
    assert evidence["tuning"]["selected_candidate_id"].startswith("lgbm-")
    assert len(evidence["tuning"]["attempts"]) == len(LIGHTGBM_SEARCH_SPACE)
    assert evidence["tuning"]["final_holdout_used_for_selection"] is False
    metadata = json.loads(
        Path(evidence["artifacts"]["metadata"]).read_text(encoding="utf-8")
    )
    assert metadata["public_test_used_for_metrics"] is False
    assert len(metadata["threshold"]["cost_sensitivity"]) == 4


def test_monitoring_accepts_only_sanitized_output_columns(tmp_path):
    frame = pd.DataFrame(
        [
            {
                "created_at": "2026-09-24T00:00:00Z",
                "transaction_amount": 10,
                "score": 0.2,
                "threshold": 0.4,
                "decision": "No",
                "latency_ms": 5,
                "model_version": "v1",
                "release_id": "r1",
                "card1": "must-not-pass",
                "SUPABASE_SERVICE_ROLE_KEY": "must-not-pass",
            }
        ]
    )
    sanitized = sanitize_monitoring_frame(frame)
    assert set(sanitized.columns).issubset(APPROVED_MONITORING_COLUMNS)
    assert "card1" not in sanitized

    reference = tmp_path / "reference.csv"
    current = tmp_path / "current.csv"
    frame.to_csv(reference, index=False)
    frame.to_csv(current, index=False)
    metrics = generate_output_monitoring_report(
        reference,
        current,
        tmp_path / "report.html",
        metrics_path=tmp_path / "metrics.json",
        min_rows=5,
    )
    assert metrics["status"] == "insufficient_data"
    assert "SUPABASE_SERVICE_ROLE_KEY" not in json.dumps(metrics)
