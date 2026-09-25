from __future__ import annotations

import json
import math
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from FraudGuard import logger
from FraudGuard.cloud.artifacts import ensure_transaction_release
from FraudGuard.cloud.persistence import (
    PredictionRecord,
    SupabasePersistence,
    utc_now_iso,
)
from FraudGuard.cloud.rate_limit import CloudRateLimiter
from FraudGuard.cloud.settings import AppSettings, load_settings
from FraudGuard.pipeline.transaction_pipeline import TransactionPipeline

settings = load_settings()
STARTUP_BUILD_TIME = datetime.now(UTC).isoformat().replace("+00:00", "Z")
BUILD_TIME_FILE = Path(__file__).resolve().parent / "build-time.txt"
FRONTEND_DIST_DIR = Path(__file__).resolve().parent / "frontend" / "dist"
FRONTEND_INDEX = FRONTEND_DIST_DIR / "index.html"
FRONTEND_ASSETS_DIR = FRONTEND_DIST_DIR / "assets"
FRONTEND_SAMPLE_CSV = FRONTEND_DIST_DIR / "sample_transactions.csv"


class TransactionBatchInput(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1)


def _safe_error_category(error: Exception) -> str:
    return error.__class__.__name__


def _log_event(level: str, event: str, **fields: Any) -> None:
    payload = {"event": event, **fields}
    getattr(logger, level)(json.dumps(payload, sort_keys=True, default=str))


def _release_id(app_settings: AppSettings, manifest: Any | None) -> str:
    if manifest is not None:
        return str(manifest.release_id)
    return app_settings.transaction_artifact_release_id or "local"


def _build_time() -> str:
    configured = os.getenv("APP_BUILD_TIME") or os.getenv("BUILD_TIME")
    if configured and configured.lower() != "unknown":
        return configured
    try:
        persisted = BUILD_TIME_FILE.read_text(encoding="utf-8").strip()
    except OSError:
        persisted = ""
    return persisted or STARTUP_BUILD_TIME


def _load_transaction_model(app: FastAPI) -> None:
    artifact_root, manifest = ensure_transaction_release(app.state.settings)
    model = TransactionPipeline(artifact_root=artifact_root)
    release_id = _release_id(app.state.settings, manifest)
    app.state.transaction_model = model
    app.state.release_id = release_id

    metadata = model.release_metadata()
    if manifest is not None:
        metadata["artifact_manifest_model_version"] = manifest.model_version
    persisted = app.state.persistence.persist_model_release(
        release_id=release_id,
        model_version=model.model_version,
        model_name=model.model_name,
        artifact_schema_version=int(model.metadata["artifact_schema_version"]),
        threshold=model.threshold,
        score_is_calibrated=model.score_is_calibrated,
        metadata=metadata,
    )
    _log_event(
        "info",
        "model_loaded",
        release_id=release_id,
        model_version=model.model_version,
        threshold=model.threshold,
        feature_count=len(model.feature_names),
        persistence_mode=app.state.persistence.mode,
        release_persisted=persisted,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.persistence = SupabasePersistence(settings)
    app.state.rate_limiter = CloudRateLimiter(settings)
    app.state.transaction_model = None
    app.state.release_id = settings.transaction_artifact_release_id or "local"
    app.state.readiness_error = None
    try:
        _load_transaction_model(app)
    except Exception as error:
        app.state.readiness_error = _safe_error_category(error)
        _log_event(
            "error",
            "model_startup_failed",
            category=app.state.readiness_error,
        )
    yield


app = FastAPI(
    title="FraudGuard API",
    description="Production-style transaction fraud scoring demonstration",
    version="2.0.0",
    lifespan=lifespan,
)

if FRONTEND_ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_ASSETS_DIR), name="assets")


def client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


@app.middleware("http")
async def add_request_context(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = request_id
    started = time.perf_counter()
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    _log_event(
        "info",
        "http_request",
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        status=response.status_code,
        latency_ms=round((time.perf_counter() - started) * 1000, 2),
    )
    return response


@app.middleware("http")
async def enforce_prediction_request_size(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/predict/transactions":
        app_settings = getattr(request.app.state, "settings", settings)
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                if int(content_length) > app_settings.max_request_bytes:
                    return _request_too_large(app_settings)
            except ValueError:
                return JSONResponse(
                    status_code=400, content={"detail": "Invalid Content-Length header"}
                )
        body = await request.body()
        if len(body) > app_settings.max_request_bytes:
            return _request_too_large(app_settings)
    return await call_next(request)


def _request_too_large(app_settings: AppSettings) -> JSONResponse:
    return JSONResponse(
        status_code=413,
        content={
            "detail": "Request body too large",
            "max_request_bytes": app_settings.max_request_bytes,
        },
    )


@app.get("/live")
async def liveness():
    return {"status": "alive", "service": "FraudGuard API", "env": settings.app_env}


@app.get("/health")
async def health():
    return await liveness()


@app.get("/version")
async def version():
    return {
        "commit_sha": (
            os.getenv("RENDER_GIT_COMMIT")
            or os.getenv("APP_COMMIT_SHA")
            or os.getenv("BUILD_COMMIT_SHA")
            or "unknown"
        ),
        "build_time": _build_time(),
    }


@app.get("/ready")
async def readiness(request: Request):
    app_settings = getattr(request.app.state, "settings", settings)
    model = getattr(request.app.state, "transaction_model", None)
    if model is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "model_loaded": False,
                "reason": getattr(
                    request.app.state,
                    "readiness_error",
                    "transaction artifacts unavailable",
                ),
            },
        )
    return {
        "status": "ready",
        "service": "FraudGuard API",
        "model_loaded": True,
        "release_id": getattr(request.app.state, "release_id", "local"),
        "model": model.readiness_metadata(),
        "persistence": request.app.state.persistence.mode,
        "rate_limit": request.app.state.rate_limiter.mode,
        "rate_limit_policy": {
            "requests": app_settings.rate_limit_requests,
            "window_seconds": app_settings.rate_limit_window_seconds,
        },
    }


@app.get("/schema/transactions")
async def transaction_schema(request: Request):
    app_settings = getattr(request.app.state, "settings", settings)
    model = getattr(request.app.state, "transaction_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Transaction model is not ready")
    return {
        "release_id": getattr(request.app.state, "release_id", "local"),
        "model_version": model.model_version,
        "model_name": model.model_name,
        "threshold": model.threshold,
        "score_is_calibrated": model.score_is_calibrated,
        "max_batch_rows": app_settings.max_batch_rows,
        "feature_count": len(model.feature_names),
        "feature_names": model.feature_names,
        "numeric_features": model.metadata.get("numeric_features", []),
        "categorical_features": model.metadata.get("categorical_features", []),
    }


@app.get("/api")
async def service_index():
    return {
        "service": "FraudGuard API",
        "status": "ok",
        "access": "anonymous_rate_limited_demo",
        "documentation": "/docs",
        "health": {"liveness": "/live", "readiness": "/ready", "version": "/version"},
        "endpoints": {
            "transaction_schema": "/schema/transactions",
            "transaction_batch_prediction": "/predict/transactions",
            "dashboard": "/dashboard",
        },
    }


@app.get("/dashboard")
async def dashboard(request: Request):
    app_settings = getattr(request.app.state, "settings", settings)
    persistence = request.app.state.persistence
    now = datetime.now(UTC)
    window_start = now - timedelta(days=app_settings.dashboard_retention_days)
    fetch_limit = app_settings.dashboard_row_limit + 1
    records = persistence.fetch_recent_predictions(
        cutoff=window_start.isoformat(), limit=fetch_limit
    )
    truncated = len(records) > app_settings.dashboard_row_limit
    records = records[: app_settings.dashboard_row_limit]
    return _dashboard_snapshot(
        records,
        persistence_mode=persistence.mode,
        window_start=window_start,
        window_end=now,
        truncated=truncated,
    )


def _dashboard_snapshot(
    records: list[dict[str, Any]],
    *,
    persistence_mode: str,
    window_start: datetime,
    window_end: datetime,
    truncated: bool,
) -> dict[str, Any]:
    flagged = [record for record in records if record.get("decision") == "Yes"]
    total = len(records)
    flagged_amount = sum(
        _finite_number(record.get("transaction_amount")) or 0.0 for record in flagged
    )
    score_total = sum(_finite_number(record.get("score")) or 0.0 for record in records)

    daily: dict[str, dict[str, Any]] = {}
    for record in records:
        day = str(record.get("created_at", ""))[:10]
        if not day:
            continue
        item = daily.setdefault(
            day, {"date": day, "transactions": 0, "flagged": 0, "fraud_rate": 0.0}
        )
        item["transactions"] += 1
        item["flagged"] += int(record.get("decision") == "Yes")
    for item in daily.values():
        item["fraud_rate"] = round((item["flagged"] / item["transactions"]) * 100, 2)

    distribution = []
    for index in range(10):
        lower = index / 10
        upper = (index + 1) / 10
        count = sum(
            1
            for record in records
            if (score := _finite_number(record.get("score"))) is not None
            and score >= lower
            and (score < upper or (index == 9 and score <= 1))
        )
        distribution.append({"score": f"{lower:.1f}", "count": count})

    recent_flagged = [
        {
            "prediction_id": str(record.get("prediction_id", ""))[:8],
            "created_at": record.get("created_at"),
            "amount": _finite_number(record.get("transaction_amount")),
            "score": _finite_number(record.get("score")) or 0.0,
            "threshold": _finite_number(record.get("threshold")) or 0.0,
            "decision": "Yes",
            "release_id": record.get("release_id") or "unknown",
        }
        for record in flagged[:5]
    ]
    latest = records[0] if records else {}
    return {
        "persistence": persistence_mode,
        "window": {
            "start": window_start.isoformat(),
            "end": window_end.isoformat(),
            "retention_days": (window_end - window_start).days,
        },
        "truncated": truncated,
        "transaction_count": total,
        "flagged_count": len(flagged),
        "fraud_rate": round((len(flagged) / total) * 100, 2) if total else 0.0,
        "flagged_amount": round(flagged_amount, 2),
        "average_score": round(score_total / total, 4) if total else 0.0,
        "threshold": _finite_number(latest.get("threshold")) or 0.0,
        "last_updated": latest.get("created_at"),
        "model_version": latest.get("model_version"),
        "release_id": latest.get("release_id"),
        "daily_volume": [daily[key] for key in sorted(daily)],
        "score_distribution": distribution,
        "recent_flagged": recent_flagged,
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


@app.get("/", include_in_schema=False)
async def frontend_index():
    if FRONTEND_INDEX.is_file():
        return FileResponse(FRONTEND_INDEX)
    return await service_index()


@app.get("/score", include_in_schema=False)
async def frontend_score_page():
    if FRONTEND_INDEX.is_file():
        return FileResponse(FRONTEND_INDEX)
    raise HTTPException(status_code=404, detail="Frontend build is not available")


@app.get("/sample_transactions.csv", include_in_schema=False)
async def frontend_sample_transactions():
    if FRONTEND_SAMPLE_CSV.is_file():
        return FileResponse(FRONTEND_SAMPLE_CSV, media_type="text/csv")
    raise HTTPException(
        status_code=404, detail="Sample transaction data is unavailable"
    )


@app.post("/predict/transactions")
async def predict_transaction_batch(request: Request, payload: TransactionBatchInput):
    app_settings = getattr(request.app.state, "settings", settings)
    if len(payload.rows) > app_settings.max_batch_rows:
        raise HTTPException(
            status_code=422,
            detail=f"Batch row count exceeds limit: {app_settings.max_batch_rows}",
        )

    limiter_result = request.app.state.rate_limiter.check(client_key(request))
    if not limiter_result.allowed:
        _log_event(
            "warning",
            "prediction_rate_limited",
            request_id=request.state.request_id,
            rate_limit_mode=limiter_result.mode,
            outcome=limiter_result.reason,
        )
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    model = getattr(request.app.state, "transaction_model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Transaction model is not ready")

    started = time.perf_counter()
    request_id = request.state.request_id
    release_id = getattr(request.app.state, "release_id", "local")
    try:
        batch_result = model.predict_rows(payload.rows)
        latency_ms = (time.perf_counter() - started) * 1000
        created_at = utc_now_iso()
        response_results = []
        records: list[PredictionRecord] = []
        for row, result in zip(payload.rows, batch_result["results"], strict=True):
            prediction_id = str(uuid4())
            records.append(
                PredictionRecord(
                    prediction_id=prediction_id,
                    request_id=request_id,
                    model_version=batch_result["model_version"],
                    release_id=release_id,
                    transaction_amount=_transaction_amount(row),
                    score=result["fraud_score"],
                    threshold=result["threshold_used"],
                    decision=result["fraud_status"],
                    score_is_calibrated=result["score_is_calibrated"],
                    latency_ms=latency_ms,
                    created_at=created_at,
                )
            )
            response_results.append(
                {"prediction_id": prediction_id, "release_id": release_id, **result}
            )

        persisted_count = request.app.state.persistence.persist_predictions(records)
        if persisted_count:
            cutoff = datetime.now(UTC) - timedelta(
                days=app_settings.dashboard_retention_days
            )
            request.app.state.persistence.delete_predictions_before(cutoff.isoformat())

        _log_event(
            "info",
            "prediction_completed",
            request_id=request_id,
            release_id=release_id,
            model_version=batch_result["model_version"],
            row_count=len(payload.rows),
            latency_ms=round(latency_ms, 2),
            persistence_mode=request.app.state.persistence.mode,
            persisted_count=persisted_count,
            rate_limit_mode=limiter_result.mode,
            status="success",
        )
        return {
            "request_id": request_id,
            "release_id": release_id,
            "model_version": batch_result["model_version"],
            "model_name": batch_result["model_name"],
            "row_count": len(response_results),
            "feature_count": batch_result["feature_count"],
            "ignored_features": batch_result["ignored_features"],
            "results": response_results,
            "latency_ms": latency_ms,
            "persistence": request.app.state.persistence.mode,
            "persisted_count": persisted_count,
            "rate_limit": limiter_result.mode,
        }
    except (TypeError, ValueError) as error:
        _log_event(
            "warning",
            "prediction_validation_failed",
            request_id=request_id,
            release_id=release_id,
            category="schema_validation",
        )
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        _log_event(
            "error",
            "prediction_failed",
            request_id=request_id,
            release_id=release_id,
            category=_safe_error_category(error),
        )
        raise HTTPException(status_code=500, detail="Internal server error") from error


def _transaction_amount(row: dict[str, Any]) -> float | None:
    for field in ("TransactionAmt", "Transaction_Amount", "amount"):
        if field in row:
            return _finite_number(row[field])
    return None


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
