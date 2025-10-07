import json
import os, sys


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from deployment.inference import model_fn, input_fn, predict_fn, output_fn



# os.environ["PYTHON"] = 



# --- Simular el entorno SageMaker ---
MODEL_DIR = "models/chronos-bolt-tiny"

# --- Simular una petición ---
request_body = json.dumps({
    "series": [[10.0, 20.0, 30.0, 40.0, 50.0]],
    "prediction_length": 3
})

print("🧠 Loading model...")
model = model_fn(MODEL_DIR)

print("📥 Preparing input...")
inputs = input_fn(request_body, "application/json")

print("⚙️ Running prediction...")
quantiles, preds = predict_fn(inputs, model)

print("📤 Producing output...")
print(output_fn(preds, "application/json"))
