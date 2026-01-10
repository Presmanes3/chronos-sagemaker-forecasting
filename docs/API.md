# API Reference

Once deployed, the model exposes the following endpoints.

## Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ping` | GET | Health check (SageMaker compatible) |
| `/health` | GET | Detailed health status |
| `/model/info` | GET | Model metadata |
| `/predict` | POST | Generate predictions |
| `/invocations` | POST | SageMaker inference endpoint |

## Request Format

```json
[
  {
    "item_id": "turbine_1",
    "timestamp": "2024-01-01 00:00:00",
    "ActivePower": 1250.5
  },
  {
    "item_id": "turbine_1",
    "timestamp": "2024-01-01 01:00:00",
    "ActivePower": 1320.8
  }
]
```

### Fields

| Field | Type | Description |
|-------|------|-------------|
| `item_id` | string | Series identifier |
| `timestamp` | string | ISO format (YYYY-MM-DD HH:MM:SS) |
| `ActivePower` | float | Target value in kW |

## Response Format

```json
{
  "predictions": {
    "columns": ["mean", "0.1", "0.5", "0.9"],
    "index": [["turbine_1", 1704153600000], ...],
    "data": [[1350.2, 1280.5, 1350.2, 1420.8], ...]
  },
  "metadata": {
    "num_series": 1,
    "num_predictions": 48,
    "request_id": "abc123"
  }
}
```

### Prediction Columns

| Column | Description |
|--------|-------------|
| `mean` | Point forecast |
| `0.1` | 10th percentile (lower bound) |
| `0.5` | Median forecast |
| `0.9` | 90th percentile (upper bound) |

## Examples

### Local Endpoint (curl)

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @src/deployment/sample_input.json
```

### Local Endpoint (Python)

```python
import requests
import json

with open('src/deployment/sample_input.json') as f:
    payload = json.load(f)

response = requests.post(
    'http://localhost:8080/predict',
    json=payload
)

result = response.json()
print(result)
```

### SageMaker Endpoint (boto3)

```python
import boto3
import json

runtime = boto3.client('sagemaker-runtime', region_name='eu-west-1')

with open('src/deployment/sample_input.json') as f:
    payload = f.read()

response = runtime.invoke_endpoint(
    EndpointName='chronos-endpoint-prod',
    ContentType='application/json',
    Body=payload
)

result = json.loads(response['Body'].read().decode())
print(result)
```

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

### Model Info

```bash
curl http://localhost:8080/model/info
```

Response:
```json
{
  "prediction_length": 48,
  "target": "ActivePower",
  "eval_metric": "RMSE",
  "model_path": "/opt/ml/model"
}
```
