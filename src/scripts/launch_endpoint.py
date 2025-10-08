import os
import boto3
import sagemaker
from sagemaker.model import Model
from dotenv import load_dotenv

load_dotenv()

role = os.environ["AWS_SAGEMAKER_ROLE_ARN"]
profile = os.environ["AWS_PROFILE"]

boto_session = boto3.Session(profile_name=profile)
sm_client = boto_session.client("sagemaker")
sm_session = sagemaker.Session(boto_session=boto_session)

MODEL_DATA = "s3://chronos-forecasting-presmanes/chronos-bolt-tiny/model.tar.gz"
ECR_IMAGE_URI = os.environ["AWS_ECR_IMAGE_URI"]  # 👈 Añádelo en tu .env
ENDPOINT_NAME = os.getenv("AWS_SAGEMAKER_ENDPOINT_NAME")
ENDPOINT_CONFIG_NAME = f"{ENDPOINT_NAME}-config"
MODEL_NAME = f"{ENDPOINT_NAME}-model"

print(f"Using model artifact from {MODEL_DATA}")
print(f"Using custom container from {ECR_IMAGE_URI}")
print(f"Deploying to endpoint: {ENDPOINT_NAME}")
print(f"Using role: {role}")

model = Model(
    image_uri           = ECR_IMAGE_URI,
    model_data          = MODEL_DATA,  
    name                = MODEL_NAME,
    role                = role,
    sagemaker_session   = sm_session,
    env={
        "HF_MODEL_ID": "amazon/chronos-bolt-tiny",
        "HF_TASK": "time-series-forecasting"
    }
)

for name in [ENDPOINT_CONFIG_NAME, ENDPOINT_NAME]:
    try:
        sm_client.describe_endpoint_config(EndpointConfigName=name)
        print(f"Endpoint config '{name}' already exists. Deleting it...")
        sm_client.delete_endpoint_config(EndpointConfigName=name)
    except sm_client.exceptions.ClientError:
        print(f"No previous endpoint config found for '{name}'.")

try:
    sm_client.describe_endpoint(EndpointName=ENDPOINT_NAME)
    print(f"Endpoint '{ENDPOINT_NAME}' already exists. Updating it...")
    update_existing = True
except sm_client.exceptions.ClientError:
    print("No previous endpoint found. Creating a new one.")
    update_existing = False

predictor = model.deploy(
    initial_instance_count  = 1,
    instance_type           = "ml.m5.large",
    endpoint_name           = ENDPOINT_NAME,
    update_endpoint         = update_existing,
    log                     = True
)

print("✅ Deployment succeeded!")
print(f"Invoke with:\n  boto3.client('sagemaker-runtime').invoke_endpoint(EndpointName='{ENDPOINT_NAME}', ContentType='application/json', Body=b'{{...}}')")
