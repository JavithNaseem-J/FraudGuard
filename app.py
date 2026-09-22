from __future__ import annotations

import time
from secrets import compare_digest
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field, field_validator

from FraudGuard import logger
from FraudGuard.cloud.artifacts import ensure_transaction_release
from FraudGuard.cloud.persistence import PredictionRecord, SupabasePersistence
from FraudGuard.cloud.rate_limit import CloudRateLimiter
from FraudGuard.cloud.settings import load_settings
from FraudGuard.pipeline.inference_pipeline import PredictionPipeline
from FraudGuard.pipeline.transaction_candidate_pipeline import (
    TransactionCandidatePipeline,
)


settings = load_settings()


def _safe_error_message(error: Exception) -> str:
    return error.__class__.__name__


def _load_baseline_predictor(app: FastAPI) -> None:
    app.state.predictor = PredictionPipeline(artifact_root=settings.model_artifact_root)
    logger.info(
        "Prediction artifacts loaded | model=%s | threshold=%.4f | persistence=%s | rate_limit=%s",
        app.state.predictor.model_version.get("version", "unknown"),
        app.state.predictor.optimal_threshold,
        app.state.persistence.mode,
        app.state.rate_limiter.mode,
    )


def _load_transaction_candidate(app: FastAPI) -> None:
    artifact_root, manifest = ensure_transaction_release(settings)
    app.state.transaction_candidate = TransactionCandidatePipeline(
        artifact_root=artifact_root
    )
    release_metadata = app.state.transaction_candidate.release_metadata()
    if manifest is not None:
        release_metadata.update(
            {
                "release_id": manifest.release_id,
                "artifact_manifest_model_version": manifest.model_version,
            }
        )
    persisted_release = app.state.persistence.persist_model_release(
        model_version=app.state.transaction_candidate.model_version,
        model_name=app.state.transaction_candidate.model_name,
        artifact_schema_version=int(
            app.state.transaction_candidate.metadata["artifact_schema_version"]
        ),
        threshold=app.state.transaction_candidate.threshold,
        score_is_calibrated=app.state.transaction_candidate.score_is_calibrated,
        metadata=release_metadata,
    )
    logger.info(
        "Transaction model loaded | version=%s | threshold=%.4f | features=%s | release_id=%s | release_persisted=%s",
        app.state.transaction_candidate.model_version,
        app.state.transaction_candidate.threshold,
        len(app.state.transaction_candidate.feature_names),
        getattr(
            manifest, "release_id", settings.transaction_artifact_release_id or "local"
        ),
        persisted_release,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.settings = settings
    app.state.persistence = SupabasePersistence(settings)
    app.state.rate_limiter = CloudRateLimiter(settings)
    app.state.predictor = None
    app.state.transaction_candidate = None
    app.state.readiness_error = None
    try:
        if settings.transaction_candidate_enabled:
            _load_transaction_candidate(app)
        else:
            _load_baseline_predictor(app)
    except Exception as error:
        app.state.readiness_error = _safe_error_message(error)
        logger.error(
            "Model startup failed | model_mode=%s | category=%s",
            settings.model_mode,
            app.state.readiness_error,
        )
    yield


app = FastAPI(
    title="FraudGuard API",
    description="Cloud-ready fraud scoring API",
    version="1.1.0",
    lifespan=lifespan,
)

current_dir = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(current_dir / "templates"))

static_dir = current_dir / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


class TransactionInput(BaseModel):
    """Validated transaction input schema with constraints."""

    transaction_type: str = Field(..., description="Type of transaction")
    device_used: str = Field(..., description="Device used for transaction")
    location: str = Field(..., min_length=2, max_length=100)
    payment_method: str = Field(..., description="Payment method")
    transaction_amount: float = Field(..., gt=0, le=1000000)
    time_of_transaction: float = Field(..., ge=0, lt=24)
    previous_fraudulent_transactions: int = Field(..., ge=0, le=100)
    account_age: int = Field(..., ge=0, le=36500)
    number_of_transactions_last_24h: int = Field(..., ge=0, le=1000)

    @field_validator("transaction_amount")
    @classmethod
    def validate_amount(cls, value: float) -> float:
        return round(value, 2)


class FeedbackInput(BaseModel):
    prediction_id: str
    confirmed_label: int = Field(..., ge=0, le=1)
    reviewer_decision: str | None = None
    feedback_source: str = "manual"


class TransactionBatchInput(BaseModel):
    rows: list[dict[str, Any]] = Field(..., min_length=1, max_length=1000)


PROTECTED_PATHS = {"/predict", "/predict/transactions", "/feedback"}


def client_key(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    if request.client:
        return request.client.host
    return "unknown"


def input_to_frame(transaction: TransactionInput) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Transaction_Type": transaction.transaction_type,
                "Device_Used": transaction.device_used,
                "Location": transaction.location,
                "Payment_Method": transaction.payment_method,
                "Transaction_Amount": transaction.transaction_amount,
                "Time_of_Transaction": transaction.time_of_transaction,
                "Previous_Fraudulent_Transactions": transaction.previous_fraudulent_transactions,
                "Account_Age": transaction.account_age,
                "Number_of_Transactions_Last_24H": transaction.number_of_transactions_last_24h,
            }
        ]
    )


def _extract_api_key(request: Request) -> str:
    header_key = request.headers.get("x-api-key", "")
    if header_key:
        return header_key.strip()
    authorization = request.headers.get("authorization", "")
    prefix = "bearer "
    if authorization.lower().startswith(prefix):
        return authorization[len(prefix) :].strip()
    return ""


def _audit_security_event(request: Request, outcome: str) -> None:
    persistence = getattr(request.app.state, "persistence", None)
    if persistence is None:
        return
    persistence.persist_audit_event(
        event_type="protected_request_denied",
        entity_id=getattr(request.state, "request_id", None),
        metadata={
            "path": request.url.path,
            "method": request.method,
            "security_outcome": outcome,
            "client": client_key(request),
        },
    )


def require_api_key(request: Request) -> None:
    app_settings = getattr(request.app.state, "settings", settings)
    if not app_settings.auth_required:
        return
    if not app_settings.fraudguard_api_key:
        logger.error("API authentication is required but no API key is configured")
        raise HTTPException(
            status_code=503,
            detail="API authentication is not configured",
        )

    supplied_key = _extract_api_key(request)
    if not supplied_key:
        _audit_security_event(request, "missing_api_key")
        logger.warning(
            "request_id=%s path=%s security_outcome=missing_api_key",
            getattr(request.state, "request_id", "unknown"),
            request.url.path,
        )
        raise HTTPException(status_code=401, detail="API key required")
    if not compare_digest(supplied_key, app_settings.fraudguard_api_key):
        _audit_security_event(request, "invalid_api_key")
        logger.warning(
            "request_id=%s path=%s security_outcome=invalid_api_key",
            getattr(request.state, "request_id", "unknown"),
            request.url.path,
        )
        raise HTTPException(status_code=403, detail="Invalid API key")


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["x-request-id"] = request_id
    return response


@app.middleware("http")
async def enforce_request_size(request: Request, call_next):
    app_settings = getattr(request.app.state, "settings", settings)
    content_length = request.headers.get("content-length")
    if content_length and app_settings.max_request_bytes > 0:
        try:
            byte_count = int(content_length)
        except ValueError:
            byte_count = 0
        if byte_count > app_settings.max_request_bytes:
            logger.warning(
                "request_id=%s path=%s security_outcome=request_too_large bytes=%s limit=%s",
                getattr(request.state, "request_id", "unknown"),
                request.url.path,
                byte_count,
                app_settings.max_request_bytes,
            )
            return JSONResponse(
                status_code=413,
                content={
                    "detail": "Request body too large",
                    "max_request_bytes": app_settings.max_request_bytes,
                },
            )
    return await call_next(request)


@app.get("/live")
async def liveness():
    return {"status": "alive", "service": "FraudGuard API", "env": settings.app_env}


@app.get("/ready")
async def readiness(request: Request):
    predictor = getattr(request.app.state, "predictor", None)
    app_settings = getattr(request.app.state, "settings", settings)
    candidate = getattr(request.app.state, "transaction_candidate", None)
    readiness_error = getattr(request.app.state, "readiness_error", None)
    if app_settings.transaction_candidate_enabled:
        if candidate is None:
            raise HTTPException(
                status_code=503,
                detail={
                    "status": "not_ready",
                    "model_mode": app_settings.model_mode,
                    "baseline_model_loaded": predictor is not None,
                    "candidate_model_loaded": False,
                    "reason": readiness_error or "transaction artifacts unavailable",
                },
            )
        candidate_metadata = candidate.readiness_metadata()
        candidate_metadata["release_id"] = (
            app_settings.rollback_release_id
            or app_settings.transaction_artifact_release_id
            or "local"
        )
        return {
            "status": "ready",
            "service": "FraudGuard API",
            "model_mode": app_settings.model_mode,
            "model_loaded": True,
            "baseline_model_loaded": False,
            "candidate_model_loaded": True,
            "threshold": candidate.threshold,
            "model_version": candidate.model_version,
            "transaction_candidate": candidate_metadata,
            "persistence": request.app.state.persistence.mode,
            "rate_limit": request.app.state.rate_limiter.mode,
        }

    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail={
                "status": "not_ready",
                "model_mode": app_settings.model_mode,
                "baseline_model_loaded": False,
                "candidate_model_loaded": candidate is not None,
                "reason": readiness_error or "prediction artifacts unavailable",
            },
        )

    candidate_metadata = (
        candidate.readiness_metadata() if candidate is not None else None
    )

    return {
        "status": "ready",
        "service": "FraudGuard API",
        "model_mode": app_settings.model_mode,
        "model_loaded": True,
        "baseline_model_loaded": True,
        "candidate_model_loaded": candidate is not None,
        "threshold": predictor.optimal_threshold,
        "model_version": predictor.model_version.get("version", "unknown"),
        "transaction_candidate": candidate_metadata,
        "persistence": request.app.state.persistence.mode,
        "rate_limit": request.app.state.rate_limiter.mode,
    }


@app.get("/health")
async def health_check(request: Request):
    return await readiness(request)


@app.get("/api/placeholder/{width}/{height}")
async def placeholder_image(width: int, height: int):
    return JSONResponse(content={"width": width, "height": height})


@app.get("/")
async def home_page(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
async def predict(request: Request, transaction: TransactionInput):
    require_api_key(request)
    app_settings = getattr(request.app.state, "settings", settings)
    if app_settings.transaction_candidate_enabled:
        raise HTTPException(
            status_code=503,
            detail="The legacy single-transaction endpoint is disabled in transaction production mode",
        )
    limiter_result = request.app.state.rate_limiter.check(client_key(request))
    if not limiter_result.allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Prediction model is not ready")

    started = time.perf_counter()
    prediction_id = str(uuid4())
    try:
        result = predictor.predict(input_to_frame(transaction))
        latency_ms = (time.perf_counter() - started) * 1000
        request_id = request.state.request_id

        record = PredictionRecord(
            prediction_id=prediction_id,
            request_id=request_id,
            model_version=result["model_version"],
            model_mode="baseline",
            score=result["fraud_score"],
            threshold=result["threshold_used"],
            decision=result["fraud_status"],
            score_is_calibrated=result["score_is_calibrated"],
            latency_ms=latency_ms,
            metadata={
                "confidence": result["confidence"],
                "model_mode": "baseline",
                "rate_limit_mode": limiter_result.mode,
            },
        )
        persisted = request.app.state.persistence.persist_prediction(record)

        logger.info(
            "prediction_id=%s request_id=%s model_mode=baseline model_version=%s decision=%s score=%.4f threshold=%.4f latency_ms=%.2f persisted=%s",
            prediction_id,
            request_id,
            result["model_version"],
            result["fraud_status"],
            result["fraud_score"],
            result["threshold_used"],
            latency_ms,
            persisted,
        )

        return {
            "prediction_id": prediction_id,
            "request_id": request_id,
            "model_mode": "baseline",
            "fraud_status": result["fraud_status"],
            "fraud_score": result["fraud_score"],
            "fraud_probability": result["fraud_probability"],
            "threshold_used": result["threshold_used"],
            "confidence": result["confidence"],
            "model_version": result["model_version"],
            "score_is_calibrated": result["score_is_calibrated"],
            "latency_ms": latency_ms,
            "persistence": request.app.state.persistence.mode,
            "rate_limit": limiter_result.mode,
            "results_url": (
                f"/results?prediction_id={prediction_id}"
                f"&fraud_status={result['fraud_status']}"
                f"&fraud_probability={result['fraud_probability']}"
                f"&threshold_used={result['threshold_used']}"
                f"&confidence={result['confidence']}"
                f"&model_version={result['model_version']}"
                f"&score_is_calibrated={result['score_is_calibrated']}"
            ),
        }
    except ValueError as error:
        logger.warning("Validation error: %s", error)
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.error("Prediction error: %s", error)
        raise HTTPException(status_code=500, detail="Internal server error") from error


@app.post("/predict/transactions")
async def predict_transaction_batch(request: Request, payload: TransactionBatchInput):
    require_api_key(request)
    app_settings = getattr(request.app.state, "settings", settings)
    if not app_settings.transaction_candidate_enabled:
        raise HTTPException(
            status_code=503,
            detail="Transaction candidate serving is disabled",
        )
    if len(payload.rows) > app_settings.max_batch_rows:
        raise HTTPException(
            status_code=422,
            detail=f"Batch row count exceeds limit: {app_settings.max_batch_rows}",
        )

    limiter_result = request.app.state.rate_limiter.check(client_key(request))
    if not limiter_result.allowed:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

    candidate = getattr(request.app.state, "transaction_candidate", None)
    if candidate is None:
        raise HTTPException(
            status_code=503,
            detail="Transaction candidate model is not ready",
        )

    started = time.perf_counter()
    request_id = request.state.request_id
    try:
        batch_result = candidate.predict_rows(payload.rows)
        latency_ms = (time.perf_counter() - started) * 1000
        response_results = []
        persisted_count = 0

        for result in batch_result["results"]:
            prediction_id = str(uuid4())
            record = PredictionRecord(
                prediction_id=prediction_id,
                request_id=request_id,
                model_version=batch_result["model_version"],
                model_mode="transaction_candidate",
                score=result["fraud_score"],
                threshold=result["threshold_used"],
                decision=result["fraud_status"],
                score_is_calibrated=result["score_is_calibrated"],
                latency_ms=latency_ms,
                metadata={
                    "row_index": result["row_index"],
                    "model_mode": "transaction_candidate",
                    "model_name": batch_result["model_name"],
                    "feature_count": batch_result["feature_count"],
                    "ignored_features": batch_result["ignored_features"],
                    "rate_limit_mode": limiter_result.mode,
                },
            )
            persisted = request.app.state.persistence.persist_prediction(record)
            persisted_count += int(persisted)
            response_results.append({"prediction_id": prediction_id, **result})

        logger.info(
            "request_id=%s model_mode=transaction_candidate rows=%s model_version=%s latency_ms=%.2f persisted=%s status=success",
            request_id,
            len(payload.rows),
            batch_result["model_version"],
            latency_ms,
            persisted_count,
        )

        return {
            "request_id": request_id,
            "model_mode": "transaction_candidate",
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
    except ValueError as error:
        logger.warning(
            "request_id=%s model_mode=transaction_candidate status=validation_error category=schema_validation",
            request_id,
        )
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        logger.error(
            "request_id=%s model_mode=transaction_candidate status=error category=prediction_error error=%s",
            request_id,
            error,
        )
        raise HTTPException(status_code=500, detail="Internal server error") from error


@app.post("/feedback")
async def submit_feedback(request: Request, feedback: FeedbackInput):
    require_api_key(request)
    persisted = request.app.state.persistence.persist_feedback(
        prediction_id=feedback.prediction_id,
        confirmed_label=feedback.confirmed_label,
        reviewer_decision=feedback.reviewer_decision,
        feedback_source=feedback.feedback_source,
        metadata={"request_id": request.state.request_id},
    )
    return {
        "prediction_id": feedback.prediction_id,
        "persisted": persisted,
        "persistence": request.app.state.persistence.mode,
    }


@app.get("/results")
async def show_results(
    request: Request,
    prediction_id: str | None = None,
    fraud_status: str | None = None,
    fraud_probability: float | None = None,
    threshold_used: float | None = None,
    confidence: str | None = None,
    model_version: str | None = None,
    score_is_calibrated: bool | None = None,
):
    if fraud_status is None or fraud_probability is None:
        return RedirectResponse(url="/")

    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "prediction_id": prediction_id or "unknown",
            "fraud_status": fraud_status,
            "fraud_probability": float(fraud_probability),
            "threshold_used": threshold_used,
            "confidence": confidence or "Medium",
            "model_version": model_version or "unknown",
            "score_is_calibrated": bool(score_is_calibrated),
        },
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=settings.port)
