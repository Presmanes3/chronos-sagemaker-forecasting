"""
Helper functions for data parsing and formatting.
"""
import json
import pandas as pd
from typing import Dict, Any

from autogluon.timeseries import TimeSeriesDataFrame
from .logger_config import setup_logger


logger = setup_logger()


def parse_input_data(body: bytes, request_id: str) -> TimeSeriesDataFrame:
    """Parse JSON request body into TimeSeriesDataFrame."""
    try:
        # Decode bytes to string
        if isinstance(body, (bytes, bytearray)):
            body_str = body.decode("utf-8")
        elif isinstance(body, str):
            body_str = body
        else:
            body_str = str(body)
        
        data = json.loads(body_str)
        
        # Handle both raw list and {"data": [...]} formats
        if isinstance(data, dict) and "data" in data:
            data = data["data"]
        
        df = pd.DataFrame(data)
        
        # Validate required columns
        if "timestamp" not in df.columns or "item_id" not in df.columns:
            raise ValueError("Input must contain 'timestamp' and 'item_id' columns")
        
        # Parse timestamps and remove timezone info
        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        
        # Convert to TimeSeriesDataFrame
        ts_df = TimeSeriesDataFrame.from_data_frame(
            df,
            id_column="item_id",
            timestamp_column="timestamp"
        )
        
        logger.info(
            "Input parsed successfully",
            extra={
                "request_id": request_id,
                "model_path": "",
                "num_rows": len(df),
                "num_series": ts_df.num_items
            }
        )
        
        return ts_df
        
    except Exception as e:
        logger.error(
            "Input parsing failed",
            extra={"request_id": request_id, "model_path": "", "error": str(e)},
            exc_info=True
        )
        raise ValueError(f"Failed to parse input data: {e}")


def format_output(predictions: TimeSeriesDataFrame, request_id: str) -> Dict[str, Any]:
    """Format predictions into JSON response."""
    try:
        output_json = predictions.to_json(orient="split")
        
        logger.info(
            "Output formatted successfully",
            extra={"request_id": request_id, "model_path": ""}
        )
        
        return {
            "predictions": json.loads(output_json),
            "metadata": {
                "num_series": predictions.num_items,
                "num_predictions": len(predictions),
                "request_id": request_id
            }
        }
        
    except Exception as e:
        logger.error(
            "Output formatting failed",
            extra={"request_id": request_id, "model_path": "", "error": str(e)},
            exc_info=True
        )
        raise RuntimeError(f"Failed to format output: {e}")
