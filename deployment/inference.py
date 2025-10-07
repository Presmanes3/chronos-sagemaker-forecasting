import os
import json
import time
import torch
from chronos import ChronosBoltPipeline


# --------------------------
# CONFIGURACIÓN GLOBAL
# --------------------------
MODEL_DIR = "/opt/ml/model"  # Aquí se monta tu modelo local


# --------------------------
# UTILIDADES
# --------------------------
def log(msg: str):
    """Helper para imprimir logs con timestamps (visible en CloudWatch o consola local)."""
    print(f"[Chronos] {time.strftime('%Y-%m-%d %H:%M:%S')} | {msg}", flush=True)


# --------------------------
# FUNCIONES DE INFERENCIA
# --------------------------
def model_fn(model_dir: str):
    """Carga el modelo Chronos desde el directorio local."""
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"❌ Modelo no encontrado en {model_dir}")

    log(f"🔹 Cargando modelo Chronos desde: {model_dir}")
    pipe = ChronosBoltPipeline.from_pretrained(model_dir, device_map="cpu")
    log("✅ Modelo cargado con éxito.")
    return pipe


def input_fn(request_body, content_type):
    """Parsea el input JSON recibido."""
    log("📥 Recibida nueva petición de inferencia")

    try:
        data = json.loads(request_body)
        log(f"🔸 Raw request: {data}")

        if "series" not in data:
            raise ValueError("Falta la clave requerida: 'series'")

        series = data["series"]
        pred_len = data.get("prediction_length", 3)

        log(f"➡️ Series length: {len(series)} | Prediction length: {pred_len}")
        return series, pred_len

    except Exception as e:
        log(f"❌ Error parsing input: {e}")
        raise


def predict_fn(data, model):
    """Realiza la inferencia."""
    start = time.time()
    series, pred_len = data

    # Asegurar formato tensor
    if isinstance(series[0], (float, int)):
        series = [series]

    series_tensor = torch.tensor(series, dtype=torch.float32)

    log(f"⚙️ Ejecutando predicción | input shape: {series_tensor.shape}")
    quantiles, out = model.predict_quantiles(series_tensor, prediction_length=pred_len)

    elapsed = time.time() - start
    log(f"✅ Predicción completada en {elapsed:.2f}s")

    # Log de ejemplo
    sample = quantiles.tolist()[0]
    log(f"🔹 Ejemplo forecast (primera serie, 3 valores): {sample[:3]}")

    return quantiles.tolist(), out.tolist()


def output_fn(prediction, accept):
    """Formatea la salida como JSON."""
    response = {"forecast": prediction}
    log(f"📤 Enviando respuesta: keys={list(response.keys())}, size={len(json.dumps(response))} bytes")
    return json.dumps(response)


# --------------------------
# MODO LOCAL / TEST
# --------------------------
if __name__ == "__main__":
    # Simular request local
    request_body = json.dumps({
        "series": [[10.0, 20.0, 30.0, 40.0, 50.0]],
        "prediction_length": 3
    })

    log("🧠 Modo local detectado. Iniciando prueba...")

    model = model_fn(MODEL_DIR)
    inputs = input_fn(request_body, "application/json")
    quantiles, preds = predict_fn(inputs, model)
    output = output_fn(preds, "application/json")

    print("\n=== 📈 OUTPUT ===")
    print(output)
    print("================\n")

    # Mantener contenedor vivo si estás debuggeando
    log("🌀 Manteniendo contenedor activo (Ctrl+C para salir)...")
    while True:
        time.sleep(60)
