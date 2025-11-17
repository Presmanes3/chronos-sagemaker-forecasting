import requests
import json
import sys
import traceback
from datetime import datetime, timedelta

# ===== CONFIG =====
BASE_URL = "http://localhost:8080"  # Local FastAPI/Docker
USE_SAGEMAKER = True                # Set True to test SageMaker
AWS_ENDPOINT_NAME = "chronos-endpoint-prod"
REGION = "eu-west-1"
# ===================


def log(msg, level="info"):
    """Minimal logger with simple emoji prefixes."""
    levels = {
        "info": "🟢",
        "warn": "🟠",
        "error": "🔴",
        "ok": "✅",
    }
    print(f"{levels.get(level, 'ℹ️')} {msg}")


def generate_payload():
    """Generate time series data formatted for Chronos inference."""
    now = datetime.now()
    timestamps = [
        (now - timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(48)
    ][::-1]

    # The model expects columns: item_id, timestamp, target
    data = [
    {"item_id": "series_1", "timestamp": ts, "ActivePower": float(i) + 0.5}
        for i, ts in enumerate(timestamps)
    ]

    return data


def test_ping():
    """Check health endpoint /ping."""
    url = (
        f"{BASE_URL}/ping"
        if not USE_SAGEMAKER
        else f"https://runtime.sagemaker.{REGION}.amazonaws.com/endpoints/{AWS_ENDPOINT_NAME}/ping"
    )
    log(f"Testing health endpoint: {url}", "info")

    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            log(f"Health OK → {response.text}", "ok")
        else:
            log(f"Unexpected status code {response.status_code}: {response.text}", "warn")
    except Exception as e:
        log(f"Health check failed: {e}", "error")


def test_invocation():
    """Test the inference endpoint (/invocations or SageMaker)."""
    payload = generate_payload()

    try:
        if USE_SAGEMAKER:
            import boto3
            runtime = boto3.client("sagemaker-runtime", region_name=REGION)

            log(f"Invoking SageMaker endpoint: {AWS_ENDPOINT_NAME}", "info")
            response = runtime.invoke_endpoint(
                EndpointName=AWS_ENDPOINT_NAME,
                ContentType="application/json",
                Body=json.dumps(payload),
            )
            status = response["ResponseMetadata"]["HTTPStatusCode"]
            body = response["Body"].read().decode("utf-8")

        else:
            url = f"{BASE_URL}/invocations"
            log(f"Invoking local endpoint: {url}", "info")
            headers = {"Content-Type": "application/json"}
            res = requests.post(url, headers=headers, data=json.dumps(payload), timeout=10)
            status, body = res.status_code, res.text

        if status == 200:
            log("Inference OK ✅", "ok")
            print(json.dumps(json.loads(body), indent=2))
        else:
            log(f"Inference failed [{status}] → {body}", "error")

    except requests.exceptions.RequestException as e:
        log(f"Network or connection error: {e}", "error")
    except Exception as e:
        log(f"Unexpected error during invocation: {e}", "error")
        traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    print("=== 🔍 SageMaker / FastAPI Inference Test ===\n")
    test_ping()
    print("-" * 80)
    test_invocation()
    print("-" * 80)
