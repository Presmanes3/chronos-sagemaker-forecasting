import os
import sys
import boto3
import sagemaker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))


from sagemaker.model import Model
from botocore.exceptions import ClientError
from dotenv import load_dotenv

from src.config import Config

load_dotenv()

# ----- Load configuration
config = Config("./config.yaml")

aws_profile       = os.getenv("AWS_PROFILE")
role_arn          = os.getenv("AWS_SAGEMAKER_ROLE_ARN")
ecr_image_uri     = os.getenv("AWS_ECR_DEPLOYMENT_IMAGE_URI")

endpoint_name     = config["sagemaker"]["endpoint_name"]
instance_type     = config["sagemaker"]["instance_type"]
instance_count    = config["sagemaker"]["instance_count"]

region            = os.getenv("AWS_REGION", "eu-west-1")

model_name        = f"{endpoint_name}-model"
config_name       = f"{endpoint_name}-config"

env_vars = {
    "HF_MODEL_ID": os.getenv("HF_MODEL_ID", "amazon/chronos-bolt-tiny"),
    "HF_TASK": "time-series-forecasting",
    "MODEL_CACHE_DIR": "/opt/ml/model",
    "SAGEMAKER_REGION": region,
}

# ----- Create sessions
session = boto3.Session(profile_name=aws_profile, region_name=region)
sm = session.client("sagemaker")
sagemaker_session = sagemaker.Session(boto_session=session)

def list_s3_production_models(bucket: str, prefix: str) -> list:
    """List all .tar.gz files in the specified S3 bucket/prefix."""
    s3 = session.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' not in response:
            return []
        models = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.tar.gz')]
        return models
    except Exception as e:
        print(f"Error listing S3 models: {e}")
        return []

def select_production_model_interactively(bucket: str, prefix: str) -> str:
    """Prompt user to select a production model from S3."""
    models = list_s3_production_models(bucket, prefix)
    
    if not models:
        print(f"\n⚠️  No production models found in s3://{bucket}/{prefix}")
        use_default = input("Use default from config.yaml? (y/n): ").strip().lower()
        if use_default == 'y':
            return config["paths"]["production_model"]
        else:
            sys.exit(1)
    
    print("\n🚀 Available production models in S3:\n")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print(f"  {len(models) + 1}. Use default from config.yaml")
    print()
    
    while True:
        try:
            choice = input(f"Select a model for deployment (1-{len(models) + 1}): ").strip()
            choice_idx = int(choice) - 1
            
            if choice_idx == len(models):
                return config["paths"]["production_model"]
            elif 0 <= choice_idx < len(models):
                return f"s3://{bucket}/{models[choice_idx]}"
            else:
                print("Invalid selection. Try again.")
        except (ValueError, KeyError):
            print("Invalid input. Please enter a number.")

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

# Get S3 configuration and select model
S3_BUCKET = config["s3"]["bucket"]
S3_PRODUCTION_PREFIX = config["s3"]["production_models"]["s3_prefix"]

# Select model interactively
s3_model_path = select_production_model_interactively(S3_BUCKET, S3_PRODUCTION_PREFIX)

print("\n🚀 Deploying with full control...")
print(f"📦 Selected model: {s3_model_path}\n")

# ----- Create or update SageMaker Endpoint
if model_exists(model_name):
    print(f"ℹ️ Model already exists: {model_name} (skipping creation)")
else:
    print(f"📦 Creating Model: {model_name}")
    model = Model(
        name                = model_name,
        image_uri           = ecr_image_uri,
        model_data          = s3_model_path,
        role                = role_arn,
        env                 = env_vars,
        sagemaker_session   = sagemaker_session,
    )
    model.create()

# ----- Create or update EndpointConfig
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

# ----- Create or update Endpoint
if endpoint_exists(endpoint_name):
    print(f"🔄 Updating existing endpoint: {endpoint_name}")

    sm.update_endpoint(
        EndpointName        = endpoint_name,
        EndpointConfigName  = config_name
    )

else:
    print(f"🚀 Creating new endpoint: {endpoint_name}")

    sm.create_endpoint(
        EndpointName        = endpoint_name,
        EndpointConfigName  = config_name
    )

print("\n✅ Deployment succeeded!")
print(f"Endpoint: {endpoint_name}")
print("You can now invoke it via boto3.sagemaker-runtime.")
