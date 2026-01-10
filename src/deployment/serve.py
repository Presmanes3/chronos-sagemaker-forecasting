"""
Chronos Forecasting API - Inference service for AutoGluon TimeSeriesPredictor.
"""

import os
import uuid
import uvicorn
from contextlib import asynccontextmanager
from typing import Optional
from pathlib import Path

from fastapi import FastAPI, Request, HTTPException, Header

from models import (
    PredictionRequest,
    PredictionResponse,
    ModelInfoResponse,
    HealthResponse
)
from predictor_service import ChronosPredictorService
from utils import parse_input_data, format_output
from logger_config import setup_logger


# ==================== Configuration ====================
MODEL_BASE_PATH = os.getenv("MODEL_DIR", "/opt/ml/model")
logger = setup_logger()


def find_model_path(base_dir: str = MODEL_BASE_PATH) -> str:
    """Find AutoGluon model directory containing predictor.pkl."""
    base_path = Path(base_dir)
    
    logger.info(
        f"Searching for model in: {base_path}",
        extra={"request_id": "startup", "search_path": str(base_path)}
    )
    
    # Check root directory first
    if (base_path / "predictor.pkl").exists():
        logger.info(
            f"Model found at root: {base_path}",
            extra={"request_id": "startup", "model_path": str(base_path)}
        )
        return str(base_path)
    
    # Search subdirectories
    for predictor_file in base_path.rglob("predictor.pkl"):
        model_dir = predictor_file.parent
        logger.info(
            f"Model found in subdirectory: {model_dir}",
            extra={"request_id": "startup", "model_path": str(model_dir)}
        )
        return str(model_dir)
    
    # Model not found - log directory structure for debugging
    logger.error(
        f"No valid AutoGluon model found in {base_path}",
        extra={"request_id": "startup", "search_path": str(base_path)}
    )
    
    if base_path.exists():
        logger.error("Directory structure:", extra={"request_id": "startup"})
        for item in sorted(base_path.rglob("*"))[:50]:
            rel_path = item.relative_to(base_path)
            item_type = "[DIR]" if item.is_dir() else "[FILE]"
            logger.error(
                f"  {item_type} {rel_path}",
                extra={"request_id": "startup"}
            )
    else:
        logger.error(
            f"Base directory does not exist: {base_path}",
            extra={"request_id": "startup"}
        )
    
    raise FileNotFoundError(
        f"No valid AutoGluon model found. Expected predictor.pkl in {base_path} or subdirectories.\n"
        f"Note: Use create_baseline_model.py to create baseline models in AutoGluon format."
    )


MODEL_PATH = None


# ==================== FastAPI Application ====================
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager - loads model at startup, cleans up at shutdown."""
    global MODEL_PATH
    
    logger.info(
        "Application starting up",
        extra={"request_id": "startup", "model_base_path": MODEL_BASE_PATH}
    )
    
    try:
        # Find model location
        MODEL_PATH = find_model_path()
        
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
    version="3.0.0",
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
            extra={"request_id": str(uuid.uuid4()), "model_path": MODEL_PATH, "error": str(e)}
        )
        raise HTTPException(status_code=500, detail="Failed to retrieve model info")


@app.post("/predict", tags=["Inference"])
async def predict(
    request: Request,
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """
    Generate time series forecasts.
    
    Accepts historical time series data and returns predictions for the configured
    forecast horizon. Input must include item_id, timestamp, and target columns.
    
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
            extra={"request_id": request_id, "model_path": MODEL_PATH, "error": str(e)}
        )
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(
            "Prediction request failed",
            extra={"request_id": request_id, "model_path": MODEL_PATH, "error": str(e)},
            exc_info=True
        )
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/invocations", tags=["Inference"])
async def invocations(
    request: Request,
    x_request_id: Optional[str] = Header(None, alias="X-Request-ID")
):
    """
    SageMaker-compatible inference endpoint.
    
    This endpoint provides backward compatibility with AWS SageMaker's default
    invocation route. It's functionally identical to /predict.
    """
    # Reuse predict endpoint logic
    return await predict(request, x_request_id)


# ==================== Application Entry Point ====================
if __name__ == "__main__":
    logger.info(
        "Starting Chronos Forecasting API server",
        extra={"request_id": "startup", "model_path": MODEL_PATH, "port": 8080}
    )
    uvicorn.run(app, host="0.0.0.0", port=8080)