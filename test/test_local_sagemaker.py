import boto3
import json
import os

from sagemaker.local import LocalSession
from sagemaker.model import Model

from dotenv import load_dotenv

load_dotenv()


# --- Configurar sesión local ---
sagemaker_session = LocalSession()
sagemaker_session.config = {"local": {"local_code": True}}

# --- Parámetros de tu entorno ---
role = os.environ["AWS_SAGEMAKER_ROLE_ARN"]
image_uri = "chronos-sagemaker:latest"
model_data = "file://./models/chronos-bolt-tiny.tar.gz" 

# --- Crear el modelo ---
model = Model(
    image_uri=image_uri,
    model_data=model_data,
    role=role,
    sagemaker_session=sagemaker_session
)

print("🚀 Desplegando el contenedor local de SageMaker...")
predictor = model.deploy(initial_instance_count=1, instance_type="local")

# --- Hacer una predicción ---
payload = {"series": [[10, 20, 30, 40, 50]], "prediction_length": 3}
response = predictor.predict(payload)

print("\n📈 Predicción:")
print(json.dumps(response, indent=2))

# --- Parar el contenedor ---
predictor.delete_endpoint()
