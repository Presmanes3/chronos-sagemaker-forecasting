import os
import json
import time
import torch
from chronos import ChronosBoltPipeline

MODEL_DIR = "/opt/ml/model"  # This is where your local model is mounted


def log(msg: str):
    """Helper to print logs with timestamps (visible in CloudWatch or local console)."""
    print(f"[Chronos] {time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)


def model_fn(model_dir: str):
    """Loads the Chronos model from the local directory."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ Model not found in {model_dir}")

    log(f"Loading Chronos model from: {model_dir}")
    pipe = ChronosBoltPipeline.from_pretrained(model_dir, device_map="cpu")
    log("Model successfully loaded.")
    return pipe

def input_fn(request_body, content_type):
    """Parses the received input JSON."""
    log("📥 Received new inference request")

    try:
        data = json.loads(request_body)
        log(f"Raw request: {data}")

        if "series" not in data:
            raise ValueError("Missing required key: 'series'")

        series = data["series"]
        pred_len = data.get("prediction_length", 3)

        log(f"Series length: {len(series)} | Prediction length: {pred_len}")
        return series, pred_len

    except Exception as e:
        log(f"❌ Error parsing input: {e}")
        raise

def predict_fn(data, model):
    """Performs inference."""
    start = time.time()
    series, pred_len = data

    # Ensure tensor format
    if isinstance(series[0], (float, int)):
        series = [series]

    series_tensor = torch.tensor(series, dtype=torch.float32)

    log(f"Running prediction | input shape: {series_tensor.shape}")
    quantiles, out = model.predict_quantiles(series_tensor, prediction_length=pred_len)

    elapsed = time.time() - start
    log(f"Prediction completed in {elapsed:.2f}s")

    # Example log
    sample = quantiles.tolist()[0]
    
    log(f"Example forecast (first series, 3 values): {sample[:3]}")

    return quantiles.tolist(), out.tolist()

def output_fn(prediction, accept):
    """Formats the output as JSON."""
    response = {"forecast": prediction}
    
    log(f"Sending response: keys={list(response.keys())}, size={len(json.dumps(response))} bytes")
    
    return json.dumps(response)

# if __name__ == "__main__":
#     request_body = json.dumps({
#         "series": [[10.0, 20.0, 30.0, 40.0, 50.0]],
#         "prediction_length": 3
#     })

#     log("🧠 Local mode detected. Starting test...")

#     model = model_fn(MODEL_DIR)
#     inputs = input_fn(request_body, "application/json")
#     quantiles, preds = predict_fn(inputs, model)
#     output = output_fn(preds, "application/json")

#     print("\n=== 📈 OUTPUT ===")
#     print(output)
#     print("================\n")

#     log("Keeping container active (Ctrl+C to exit)...")
#     while True:
#         time.sleep(60)
