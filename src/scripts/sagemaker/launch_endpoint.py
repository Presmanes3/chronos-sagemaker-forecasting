import os
import boto3
import sagemaker
from sagemaker.model import Model
from botocore.exceptions import ClientError
from dotenv import load_dotenv

load_dotenv()

aws_profile       = os.getenv("AWS_PROFILE")
role_arn          = os.getenv("AWS_SAGEMAKER_ROLE_ARN")
ecr_image_uri     = os.getenv("AWS_ECR_DEPLOYMENT_IMAGE_URI")
s3_model_path     = os.getenv("PRODUCTION_MODEL_PATH")
endpoint_name     = os.getenv("AWS_SAGEMAKER_ENDPOINT_NAME")
instance_type     = os.getenv("AWS_SAGEMAKER_INSTANCE_TYPE", "ml.m5.large")
instance_count    = int(os.getenv("AWS_SAGEMAKER_INSTANCE_COUNT", "1"))
region            = os.getenv("AWS_REGION", "eu-west-1")

model_name        = f"{endpoint_name}-model"
config_name       = f"{endpoint_name}-config"

env_vars = {
    "HF_MODEL_ID": os.getenv("HF_MODEL_ID", "amazon/chronos-bolt-tiny"),
    "HF_TASK": "time-series-forecasting",
    "MODEL_CACHE_DIR": "/opt/ml/model",
    "SAGEMAKER_REGION": region,
}

session = boto3.Session(profile_name=aws_profile, region_name=region)
sm = session.client("sagemaker")
sagemaker_session = sagemaker.Session(boto_session=session)

def model_exists(name):
    try:
        sm.describe_model(ModelName=name)
        return True
    except ClientError:
        return False

def config_exists(name):
    try:
        sm.describe_endpoint_config(EndpointConfigName=name)
        return True
    except ClientError:
        return False

def endpoint_exists(name):
    try:
        sm.describe_endpoint(EndpointName=name)
        return True
    except ClientError:
        return False

print("🚀 Deploying with full control...")

# ------------------------------------------------------
# Step 1 — Create or update Model
# ------------------------------------------------------
if model_exists(model_name):
    print(f"ℹ️ Model already exists: {model_name} (skipping creation)")
else:
    print(f"📦 Creating Model: {model_name}")
    model = Model(
        name=model_name,
        image_uri=ecr_image_uri,
        model_data=s3_model_path,
        role=role_arn,
        env=env_vars,
        sagemaker_session=sagemaker_session,
    )
    model.create()

# ------------------------------------------------------
# Step 2 — Create Endpoint Config
# ------------------------------------------------------
if config_exists(config_name):
    print(f"ℹ️ EndpointConfig already exists: {config_name} (recreating)")

    sm.delete_endpoint_config(EndpointConfigName=config_name)

print(f"⚙️ Creating EndpointConfig: {config_name}")

sm.create_endpoint_config(
    EndpointConfigName=config_name,
    ProductionVariants=[
        {
            "VariantName": "AllTraffic",
            "ModelName": model_name,
            "InitialInstanceCount": instance_count,
            "InstanceType": instance_type,
        }
    ]
)

# ------------------------------------------------------
# Step 3 — Create or update Endpoint
# ------------------------------------------------------
if endpoint_exists(endpoint_name):
    print(f"🔄 Updating existing endpoint: {endpoint_name}")

    sm.update_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name
    )

else:
    print(f"🚀 Creating new endpoint: {endpoint_name}")

    sm.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=config_name
    )

print("\n✅ Deployment succeeded!")
print(f"Endpoint: {endpoint_name}")
print("You can now invoke it via boto3.sagemaker-runtime.")
