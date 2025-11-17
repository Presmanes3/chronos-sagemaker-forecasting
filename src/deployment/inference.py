import os
import json
import time
import shutil
import tempfile
import pandas as pd
from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def log(message: str, level: str = "INFO"):
    """Utility logger with timestamps."""
    print(f"[Chronos] {time.strftime('%Y-%m-%d %H:%M:%S')} [{level}] {message}", flush=True)


def model_fn(model_dir: str):
    """Load the trained AutoGluon model into a writable /tmp directory."""
    try:
        log(f"Loading model from read-only dir: {model_dir}")

        # ✅ Create a temporary writable directory inside /tmp
        tmp_model_dir = os.path.join(tempfile.gettempdir(), "chronos_model_copy")
        if os.path.exists(tmp_model_dir):
            shutil.rmtree(tmp_model_dir)
        shutil.copytree(model_dir, tmp_model_dir)

        log(f"Copied model to writable dir: {tmp_model_dir}")

        # ✅ Load the model from /tmp (not /opt/ml/model)
        predictor = TimeSeriesPredictor.load(tmp_model_dir)

        # ✅ Force paths inside model to stay in /tmp
        predictor.path = tmp_model_dir
        predictor._learner.path = tmp_model_dir
        predictor._trainer.path = os.path.join(tmp_model_dir, "models")

        log("Model loaded successfully ✅ — all paths redirected to /tmp.")
        return predictor

    except Exception as e:
        log(f"Failed to load model: {e}", "ERROR")
        raise


def input_fn(request_body, content_type: str):
    """Parse JSON input into TimeSeriesDataFrame."""
    try:
        if isinstance(request_body, (bytes, bytearray)):
            request_body = request_body.decode("utf-8")

        data = json.loads(request_body)
        df = pd.DataFrame(data)

        if "timestamp" not in df.columns or "item_id" not in df.columns:
            raise ValueError("Input JSON must include 'timestamp' and 'item_id' columns.")

        df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
        ts_df = TimeSeriesDataFrame.from_data_frame(df, id_column="item_id", timestamp_column="timestamp")

        log(f"Input parsed successfully with {len(df)} rows.")
        return ts_df

    except Exception as e:
        log(f"Error parsing input: {e}", "ERROR")
        raise


def predict_fn(ts_df: TimeSeriesDataFrame, model: TimeSeriesPredictor):
    """Run inference safely in a writable /tmp context."""
    try:
        tmp_dir = tempfile.mkdtemp(dir="/tmp")
        os.chdir(tmp_dir)
        predictions = model.predict(ts_df)
        log(f"Prediction completed for {len(ts_df)} series.")
        return predictions
    except Exception as e:
        log(f"Prediction failed: {e}", "ERROR")
        raise
    finally:
        os.chdir("/")


def output_fn(predictions, accept: str):
    """Format predictions into JSON for HTTP response."""
    try:
        output_json = predictions.to_json(orient="split")
        log("Output formatted successfully.")
        return {"predictions": json.loads(output_json)}
    except Exception as e:
        log(f"Error formatting output: {e}", "ERROR")
        return {"error": str(e)}
