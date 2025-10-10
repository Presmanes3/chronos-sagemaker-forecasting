import os
import boto3
import sagemaker

from sagemaker.model import Model
from dotenv import load_dotenv

# ------------------------------------------------------
# Load environment variables
# ------------------------------------------------------
load_dotenv()

# Required variables
aws_profile       = os.getenv("AWS_PROFILE")
role_arn          = os.getenv("AWS_SAGEMAKER_ROLE_ARN")
ecr_image_uri     = os.getenv("AWS_ECR_DEPLOYMENT_IMAGE_URI")
s3_model_path     = os.getenv("PRODUCTION_MODEL_PATH")  # S3 path to model.tar.gz
endpoint_name     = os.getenv("AWS_SAGEMAKER_ENDPOINT_NAME")
instance_type     = os.getenv("AWS_SAGEMAKER_INSTANCE_TYPE", "ml.m5.large")
instance_count    = int(os.getenv("AWS_SAGEMAKER_INSTANCE_COUNT", "1"))

# Optional environment vars for the model
model_env_vars = {
    "HF_MODEL_ID": os.getenv("HF_MODEL_ID", "amazon/chronos-bolt-tiny"),
    "HF_TASK": os.getenv("HF_TASK", "time-series-forecasting"),
    "MODEL_CACHE_DIR": os.getenv("MODEL_CACHE_DIR", "/opt/ml/model"),
    "SAGEMAKER_REGION": os.getenv("AWS_REGION", "eu-west-1"),
}

# Validate required ones
missing = [
    k for k, v in {
        "AWS_PROFILE": aws_profile,
        "AWS_SAGEMAKER_ROLE_ARN": role_arn,
        "AWS_ECR_DEPLOYMENT_IMAGE_URI": ecr_image_uri,
        "AWS_S3_MODEL_PATH": s3_model_path,
        "AWS_SAGEMAKER_ENDPOINT_NAME": endpoint_name,
    }.items() if v is None
]
if missing:
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

# ------------------------------------------------------
# SageMaker setup
# ------------------------------------------------------
boto_session = boto3.Session(profile_name=aws_profile)
sagemaker_session = sagemaker.Session(boto_session=boto_session)

print("🚀 Starting endpoint deployment...")
print(f"Model artifact: {s3_model_path}")
print(f"Image URI:      {ecr_image_uri}")
print(f"Endpoint:       {endpoint_name}")
print(f"Role:           {role_arn}")

# ------------------------------------------------------
# Create SageMaker model and deploy
# ------------------------------------------------------
model = Model(
    image_uri           = ecr_image_uri,
    model_data          = s3_model_path,
    role                = role_arn,
    sagemaker_session   = sagemaker_session,
    env                 = model_env_vars,
)

predictor = model.deploy(
    initial_instance_count  = instance_count,
    instance_type           = instance_type,
    endpoint_name           = endpoint_name,
    log                     = True,
)

# ------------------------------------------------------
# Summary
# ------------------------------------------------------
print("\n✅ Deployment succeeded!")
print(f"Endpoint name: {endpoint_name}")
print(f"Invoke example:\n")
print(f"boto3.client('sagemaker-runtime').invoke_endpoint(")
print(f"    EndpointName='{endpoint_name}',")
print(f"    ContentType='application/json',")
print(f"    Body=b'{{\"instances\": [1,2,3,4]}}')")
