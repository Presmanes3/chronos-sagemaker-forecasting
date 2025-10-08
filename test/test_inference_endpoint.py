import os
import json
import boto3
import numpy as np
from dotenv import load_dotenv

# -----------------------------
# Load AWS credentials
# -----------------------------
load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE")
ENDPOINT_NAME = os.getenv("AWS_SAGEMAKER_ENDPOINT_NAME")

# Use your configured profile
boto_session = boto3.Session(profile_name=AWS_PROFILE)
runtime_client = boto_session.client("sagemaker-runtime")

# -----------------------------
# Prepare input data
# -----------------------------
# Example input: a normalized time series of 5 points
series = [10.0, 20.0, 30.0, 40.0, 50.0]
payload = {
    "series": series,
    "prediction_length": 3
}

print("📤 Sending request to SageMaker endpoint...")

# -----------------------------
# Invoke endpoint
# -----------------------------
response = runtime_client.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType="application/json",
    Body=json.dumps(payload)
)

# -----------------------------
# Parse response
# -----------------------------
result = json.loads(response["Body"].read().decode("utf-8"))
print("✅ Inference result:")
print(json.dumps(result, indent=2))
