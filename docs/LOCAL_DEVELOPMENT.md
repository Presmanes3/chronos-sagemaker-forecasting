# Local Development

Guide for running the inference service locally.

## Prerequisites

- Python 3.10+
- Docker (optional)
- Trained model artifacts

## Option 1: Run with Python

### Install dependencies

```bash
cd src/deployment
pip install -r requirements.txt
```

### Set environment variables

```bash
export MODEL_DIR=/path/to/model/artifacts
```

### Start server

```bash
python serve.py
```

Server runs at `http://localhost:8080`

## Option 2: Run with Docker

### Build image

```bash
docker build -t chronos-api -f src/deployment/dockerfile src/deployment
```

### Run container

```bash
docker run -p 8080:8080 \
  -v /path/to/model:/opt/ml/model \
  chronos-api
```

## Option 3: Run with Docker Compose

```bash
docker-compose up
```

## Testing the API

### Health check

```bash
curl http://localhost:8080/ping
```

### Get model info

```bash
curl http://localhost:8080/model/info
```

### Make prediction

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d @src/deployment/sample_input.json
```

## Project Structure

```
src/deployment/
├── serve.py              # FastAPI application entry point
├── predictor_service.py  # Model loading and prediction logic
├── models.py             # Pydantic request/response schemas
├── utils.py              # Data parsing utilities
├── logger_config.py      # Logging configuration
├── dockerfile            # Container definition
├── requirements.txt      # Python dependencies
└── sample_input.json     # Example request payload
```

## Configuration

The service uses the following environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_DIR` | `/opt/ml/model` | Path to model artifacts |
| `PORT` | `8080` | Server port |

## Debugging

### Enable debug logging

Modify `logger_config.py`:

```python
logger.setLevel(logging.DEBUG)
```

### Check model structure

The model directory must contain:
- `predictor.pkl` - AutoGluon predictor
- `learner.pkl` - AutoGluon learner
- `models/` - Model weights
