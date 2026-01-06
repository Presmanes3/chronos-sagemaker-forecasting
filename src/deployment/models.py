"""
Pydantic models for request/response validation.
"""
import pandas as pd
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field, validator


class TimeSeriesDataPoint(BaseModel):
    """Single time series observation."""
    item_id: str = Field(..., description="Series identifier")
    timestamp: str = Field(..., description="ISO format timestamp (YYYY-MM-DD HH:MM:SS)")
    ActivePower: float = Field(..., description="Active power value in kW")
    
    @validator("timestamp")
    def validate_timestamp(cls, v):
        """Ensure timestamp can be parsed."""
        try:
            pd.to_datetime(v)
            return v
        except Exception as e:
            raise ValueError(f"Invalid timestamp format: {v}. Expected ISO format (YYYY-MM-DD HH:MM:SS)")


class PredictionRequest(BaseModel):
    """Request payload for /predict endpoint."""
    data: List[TimeSeriesDataPoint] = Field(..., description="Time series historical data")
    prediction_length: Optional[int] = Field(None, description="Override prediction horizon (optional)")
    
    class Config:
        schema_extra = {
            "example": {
                "data": [
                    {"item_id": "turbine_1", "timestamp": "2024-01-01 00:00:00", "ActivePower": 1250.5},
                    {"item_id": "turbine_1", "timestamp": "2024-01-01 01:00:00", "ActivePower": 1320.8}
                ]
            }
        }


class PredictionResponse(BaseModel):
    """Response payload for successful predictions."""
    predictions: Dict[str, Any] = Field(..., description="TimeSeriesDataFrame in split-oriented JSON")
    metadata: Dict[str, Any] = Field(..., description="Prediction metadata")


class ModelInfoResponse(BaseModel):
    """Model metadata and configuration."""
    prediction_length: int = Field(..., description="Default forecast horizon (steps)")
    target: str = Field(..., description="Target variable name")
    eval_metric: Optional[str] = Field(None, description="Evaluation metric")
    model_path: str = Field(..., description="Model storage path")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service health status")
    model_loaded: bool = Field(..., description="Whether model is loaded")
