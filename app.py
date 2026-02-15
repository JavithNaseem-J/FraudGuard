from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator
from typing import Literal
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import os
import pandas as pd
import json
import uvicorn
from FraudGuard.pipeline.inference_pipeline import PredictionPipeline
from FraudGuard import logger

# Initialize FastAPI app
app = FastAPI(
    title="FraudGuard API",
    description="Real-time fraud detection system",
    version="1.0.0"
)

# Initialize rate limiter
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Setup templates and static files
current_dir = os.path.dirname(os.path.abspath(__file__))
template_dir = os.path.join(current_dir, 'templates')
templates = Jinja2Templates(directory=template_dir)

static_dir = os.path.join(current_dir, 'static')
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")


# Pydantic model for input validation
class TransactionInput(BaseModel):
    """Validated transaction input schema with constraints"""
    
    transaction_type: str = Field(..., description="Type of transaction")
    device_used: str = Field(..., description="Device used for transaction")
    location: str = Field(..., min_length=2, max_length=100, description="Transaction location")
    payment_method: str = Field(..., description="Payment method")
    
    transaction_amount: float = Field(
        ..., 
        gt=0,  # Must be positive
        le=1000000,  # Max 1M
        description="Transaction amount"
    )
    
    time_of_transaction: float = Field(
        ...,
        ge=0,
        lt=24,
        description="Hour of day (0-23.99)"
    )
    
    previous_fraudulent_transactions: int = Field(
        ...,
        ge=0,
        le=100,
        description="Count of previous fraud cases"
    )
    
    account_age: int = Field(
        ...,
        ge=0,
        le=36500,  # ~100 years in days
        description="Account age in days"
    )
    
    number_of_transactions_last_24h: int = Field(
        ...,
        ge=0,
        le=1000,
        description="Transaction count in last 24 hours"
    )
    
    @field_validator('transaction_amount')
    @classmethod
    def validate_amount(cls, v):
        """Validate and round transaction amount"""
        if v <= 0:
            raise ValueError('Transaction amount must be positive')
        return round(v, 2)
    
    @field_validator('time_of_transaction')
    @classmethod
    def validate_time(cls, v):
        """Validate time is within 24-hour range"""
        if not 0 <= v < 24:
            raise ValueError('Time must be between 0 and 23.99')
        return v


@app.get("/health")
async def health_check():
    """
    Robust health check that verifies model is loaded and functional.
    Returns 503 if service is unhealthy.
    """
    try:
        # Initialize pipeline to verify all artifacts are loaded
        pipeline = PredictionPipeline()
        
        # Verify critical artifacts exist
        required_files = [
            pipeline.model_path,
            pipeline.preprocessor_path,
            pipeline.label_encoders_path,
            pipeline.threshold_path
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                raise FileNotFoundError(f"Critical artifact missing: {file_path}")
        
        return {
            "status": "healthy",
            "service": "FraudGuard API",
            "model_loaded": True,
            "threshold": pipeline.optimal_threshold,
            "model_version": pipeline.model_version.get("version", "unknown")
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        raise HTTPException(
            status_code=503, 
            detail=f"Service unhealthy: {str(e)}"
        )


@app.get("/api/placeholder/{width}/{height}")
async def placeholder_image(width: int, height: int):
    """Placeholder image endpoint for frontend"""
    return JSONResponse(content={"width": width, "height": height})


@app.get("/")
async def home_page(request: Request):
    """Render home page"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/predict")
@limiter.limit("20/minute")  # Max 20 requests per minute per IP
async def predict(request: Request, transaction: TransactionInput):
    """
    Fraud prediction endpoint with input validation and rate limiting.
    
    Args:
        request: FastAPI request object
        transaction: Validated transaction input
        
    Returns:
        JSON response with fraud prediction results
    """
    try:
        # Convert Pydantic model to DataFrame
        input_df = pd.DataFrame([{
            'Transaction_Type': transaction.transaction_type,
            'Device_Used': transaction.device_used,
            'Location': transaction.location,
            'Payment_Method': transaction.payment_method,
            'Transaction_Amount': transaction.transaction_amount,
            'Time_of_Transaction': transaction.time_of_transaction,
            'Previous_Fraudulent_Transactions': transaction.previous_fraudulent_transactions,
            'Account_Age': transaction.account_age,
            'Number_of_Transactions_Last_24H': transaction.number_of_transactions_last_24h
        }])

        logger.info(f"Processing prediction request from {request.client.host}")

        # Initialize pipeline and make prediction
        pipeline = PredictionPipeline()
        result = pipeline.predict(input_df)
        
        fraud_status = result['fraud_status']
        fraud_probability = result['fraud_probability']
        threshold_used = result.get('threshold_used', 0.25)
        confidence = result.get('confidence', 'Medium')
        model_version = result.get('model_version', 'unknown')

        logger.info(
            f"Prediction: {fraud_status} | "
            f"Probability: {fraud_probability:.4f} | "
            f"Threshold: {threshold_used} | "
            f"Model: v{model_version}"
        )

        # Prepare data for results page
        data = transaction.dict()
        encoded_data = json.dumps(data)

        return JSONResponse(content={
            "redirect": (
                f"/results?"
                f"fraud_status={fraud_status}&"
                f"fraud_probability={fraud_probability}&"
                f"threshold_used={threshold_used}&"
                f"confidence={confidence}&"
                f"model_version={model_version}&"
                f"data={encoded_data}"
            )
        })

    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return JSONResponse(
            content={"error": f"Validation error: {str(e)}"}, 
            status_code=422
        )
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return JSONResponse(
            content={"error": "Internal server error"}, 
            status_code=500
        )


@app.get("/results")
async def show_results(
    request: Request, 
    fraud_status: str = None, 
    fraud_probability: float = None,
    threshold_used: float = None,
    confidence: str = None,
    model_version: str = None,
    data: str = None
):
    """
    Display prediction results page.
    
    Args:
        request: FastAPI request object
        fraud_status: Prediction result (Yes/No)
        fraud_probability: Fraud probability score
        threshold_used: Decision threshold used
        confidence: Prediction confidence level
        model_version: Model version used for prediction
        data: JSON-encoded input data
        
    Returns:
        Rendered results template
    """
    try:
        if fraud_status is None or fraud_probability is None or data is None:
            logger.warning("Missing required parameters for results page")
            return RedirectResponse(url="/")

        input_data = json.loads(data)

        return templates.TemplateResponse("result.html", {
            "request": request,
            "fraud_status": fraud_status,
            "fraud_probability": float(fraud_probability),
            "threshold_used": float(threshold_used) if threshold_used else 0.25,
            "confidence": confidence or "Medium",
            "model_version": model_version or "unknown",
            "transaction_data": input_data
        })
    except Exception as e:
        logger.error(f"Error in results handler: {str(e)}")
        return RedirectResponse(url="/")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)