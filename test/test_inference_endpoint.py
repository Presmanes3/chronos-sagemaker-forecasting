import boto3
import pandas as pd
import json

# Crear cliente SageMaker Runtime
runtime = boto3.client("sagemaker-runtime", region_name="eu-west-1")

# Crear un dataframe similar al de entrenamiento
data = pd.DataFrame({
    "item_id": ["series_1"] * 10,
    "timestamp": pd.date_range("2025-11-01", periods=10, freq="10min"),
    "ActivePower": [120, 122, 121, 119, 118, 121, 123, 125, 124, 126],
})

# Convertir a JSON
payload = data.to_json(orient="records")

# Llamar al endpoint
response = runtime.invoke_endpoint(
    EndpointName="chronos-forecasting-endpoint",
    ContentType="application/json",
    Body=payload
)

# Leer respuesta
result = json.loads(response["Body"].read().decode("utf-8"))
print(result)
