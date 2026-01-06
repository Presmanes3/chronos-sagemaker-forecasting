"""
Chronos Forecasting API - Production-ready inference service for AutoGluon TimeSeriesPredictor.

This FastAPI application provides a production-ready inference endpoint for time series forecasting
using Amazon Chronos models fine-tuned with AutoGluon. It includes:
- Structured JSON logging for CloudWatch
- Model metadata validation
- RESTful endpoints with Pydantic validation
- SageMaker compatibility (/ping, /invocations)
"""

import os
import sys
import uuid
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, Request, HTTPException, Header
from fastapi.responses import JSONResponse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from .models import (
    PredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse
)
from .predictor_service import ChronosPredictorService
from .utils import parse_input_data, format_output
from .logger_config import setup_logger


# ==================== Configuration ====================
MODEL_PATH = "/opt/ml/model/fine_tuned"  # SageMaker model mount point
logger = setup_logger()


# ==================== FastAPI Application ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - loads model at startup, cleans up at shutdown."""
    logger.info(
        "Application starting up",
        extra={"request_id": "startup", "model_path": MODEL_PATH}
    )
    
    try:
        # Initialize predictor service
        service = ChronosPredictorService()
        service.load_model(MODEL_PATH)
        app.state.predictor_service = service
        
        logger.info(
            "Application ready",
            extra={"request_id": "startup", "model_path": MODEL_PATH}
        )
        
    except Exception as e:
        logger.error(
            "Startup failed",
            extra={"request_id": "startup", "model_path": MODEL_PATH, "error": str(e)},
            exc_info=True
        )
        raise
    
    yield  # Application runs
    
    logger.info(
        "Application shutting down",
        extra={"request_id": "shutdown", "model_path": MODEL_PATH}
    )


app = FastAPI(
    title="Chronos Forecasting API",
    description="Production-ready time series forecasting API using Amazon Chronos + AutoGluon",
    version="2.0.0",
    lifespan=lifespan
)


# ==================== API Endpoints ====================
@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """
    Enhanced health check endpoint.
    
    Returns service status and model loading state.
    """
    model_loaded = hasattr(app.state, "predictor_service") and app.state.predictor_service.predictor is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded
    }


@app.get("/ping", tags=["Health"])
async def ping():
    """
    SageMaker-compatible health check endpoint.
    
    Required by AWS SageMaker for container health monitoring.
    """
    return {"status": "ok"}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """
    Get loaded model metadata and configuration.
    
    Returns prediction length, target variable, evaluation metric, and model path.
    """
    try:
        service: ChronosPredictorService = app.state.predictor_service
        info = service.get_info()
        
        return info
        
    except Exception as e:
        logger.error(
            "Failed to retrieve model info",
            extra={"request_id": str(uuid.uuid4()), "model_path": "", "error": str(e)}
        )
        raise HTTPException(status_code=500, detail=f"Failed to retrieve model info: {e}")


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(
    request: Request,
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """
    Generate time series forecasts.
    
    Accepts historical time series data and returns predictions for the configured
    forecast horizon. Input must include item_id, timestamp, and ActivePower columns.
    
    **Request Body:** JSON array of time series observations
    
    **Response:** Predictions in TimeSeriesDataFrame split-oriented JSON format
    """
    request_id = x_request_id or str(uuid.uuid4())
    
    try:
        service: ChronosPredictorService = app.state.predictor_service
        
        # Parse input
        body = await request.body()
        ts_df = parse_input_data(body, request_id)
        
        # Execute prediction
        predictions = service.predict(ts_df, request_id)
        
        # Format output
        result = format_output(predictions, request_id)
        
        return result
        
    except ValueError as e:
        logger.warning(
            "Invalid input data",
            extra={"request_id": request_id, "model_path": "", "error": str(e)}
        )
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(
            "Prediction request failed",
            extra={"request_id": request_id, "model_path": "", "error": str(e)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")


@app.post("/invocations", tags=["SageMaker Compatibility"])
async def invocations(
    request: Request,
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """
    SageMaker-compatible inference endpoint.
    
    This endpoint provides backward compatibility with AWS SageMaker's default
    invocation route. It's functionally identical to /predict.
    """
    request_id = x_request_id or str(uuid.uuid4())
    
    try:
        service: ChronosPredictorService = app.state.predictor_service
        
        # Parse input
        body = await request.body()
        ts_df = parse_input_data(body, request_id)
        
        # Execute prediction
        predictions = service.predict(ts_df, request_id)
        
        # Format output
        result = format_output(predictions, request_id)
        
        return JSONResponse(content=result, status_code=200)
        
    except ValueError as e:
        logger.warning(
            "Invalid input data",
            extra={"request_id": request_id, "model_path": "", "error": str(e)}
        )
        return JSONResponse(content={"error": str(e)}, status_code=400)
        
    except Exception as e:
        logger.error(
            "Inference request failed",
            extra={"request_id": request_id, "model_path": "", "error": str(e)},
            exc_info=True
        )
        return JSONResponse(content={"error": str(e)}, status_code=500)


# ==================== Application Entry Point ====================
if __name__ == "__main__":
    logger.info(
        "Starting Chronos Forecasting API server",
        extra={"request_id": "startup", "model_path": MODEL_PATH, "port": 8080}
    )
    uvicorn.run(app, host="0.0.0.0", port=8080)
