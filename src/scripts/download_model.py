from chronos import ChronosBoltPipeline
from pathlib import Path

MODEL_NAME = "amazon/chronos-bolt-tiny"
MODEL_DIR = Path("models/chronos-bolt-tiny")

MODEL_DIR.mkdir(parents=True, exist_ok=True)

print(f"Downloading model '{MODEL_NAME}' to {MODEL_DIR}...")

pipe = ChronosBoltPipeline.from_pretrained(MODEL_NAME)
pipe.model.save_pretrained(MODEL_DIR)

print("Model and tokenizer saved successfully!")
