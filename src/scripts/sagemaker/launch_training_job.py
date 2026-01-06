"""
This script launches a training job in SageMaker using a custom Docker container hosted in ECR.
Requires the following environment variables:

The training container Dockerfile is located in src/training.
"""


import os
import sys
import boto3
import sagemaker

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.config import Config

from dotenv import load_dotenv

load_dotenv()



# ----- Create sessions
boto3_session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))

session = sagemaker.Session(boto_session=boto3_session)

# ----- Load configuration
config = Config("./config.yaml")

def list_s3_models(bucket: str, prefix: str) -> list:
    """List all .tar.gz files in the specified S3 bucket/prefix."""
    s3 = boto3_session.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' not in response:
            return []
        models = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.tar.gz')]
        return models
    except Exception as e:
        print(f"Error listing S3 models: {e}")
        return []

def select_model_interactively(bucket: str, prefix: str) -> str:
    """Prompt user to select a model from S3."""
    models = list_s3_models(bucket, prefix)
    
    if not models:
        print(f"\n⚠️  No models found in s3://{bucket}/{prefix}")
        use_default = input("Use default from config.yaml? (y/n): ").strip().lower()
        if use_default == 'y':
            return config["paths"]["base_model"]
        else:
            sys.exit(1)
    
    print("\n📦 Available base models in S3:\n")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print(f"  {len(models) + 1}. Use default from config.yaml")
    print()
    
    while True:
        try:
            choice = input(f"Select a model (1-{len(models) + 1}): ").strip()
            choice_idx = int(choice) - 1
            
            if choice_idx == len(models):
                return config["paths"]["base_model"]
            elif 0 <= choice_idx < len(models):
                return f"s3://{bucket}/{models[choice_idx]}"
            else:
                print("Invalid selection. Try again.")
        except (ValueError, KeyError):
            print("Invalid input. Please enter a number.")

# Get S3 configuration
S3_BUCKET = config["s3"]["bucket"]
S3_MODELS_PREFIX = config["s3"]["upload"]["models"]["s3_prefix"]

# Select model interactively
BASE_MODEL_PATH = select_model_interactively(S3_BUCKET, S3_MODELS_PREFIX)

TRAINING_DATA_PATH  = config["paths"]["training_data"]
TUNED_MODEL_PATH    = config["paths"]["production_model"]

AWS_PROFILE         = os.getenv("AWS_PROFILE")
TRAINING_LIMIT_TIME =  str(config["training"]["limit_time"])

ECR_URI             = os.getenv("AWS_ECR_TRAINING_IMAGE_URI")
ROLE                = os.getenv("AWS_SAGEMAKER_ROLE_ARN")

if not BASE_MODEL_PATH or not TRAINING_DATA_PATH or not TUNED_MODEL_PATH or not ECR_URI or not ROLE:
    missing = [
        k for k, v in {
            "BASE_MODEL_PATH": BASE_MODEL_PATH,
            "TRAINING_DATA_PATH": TRAINING_DATA_PATH,
            "TUNED_MODEL_PATH": TUNED_MODEL_PATH,
            "AWS_ECR_TRAINING_IMAGE_URI": ECR_URI,
            "AWS_SAGEMAKER_ROLE_ARN": ROLE,
        }.items() if v is None
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

print("=" * 80)
print("SAGEMAKER TRAINING JOB CONFIGURATION")
print("=" * 80)
print("\n📦 Model Paths:")
print(f"  • Base Model:        {BASE_MODEL_PATH}")
print(f"  • Tuned Model:       {TUNED_MODEL_PATH}")
print("\n📊 Data Paths:")
print(f"  • Training Data:     {TRAINING_DATA_PATH}")
print("\n⚙️  Training Configuration:")
print(f"  • Time Limit:        {TRAINING_LIMIT_TIME} seconds")
print(f"  • Instance Type:     ml.m5.large")
print(f"  • Instance Count:    1")
print("\n🔧 AWS Resources:")
print(f"  • Profile:           {AWS_PROFILE}")
print(f"  • ECR Image URI:     {ECR_URI}")
print(f"  • IAM Role ARN:      {ROLE}")
print("\n" + "=" * 80)
print("🚀 Launching training job...")
print("=" * 80 + "\n")

estimator = sagemaker.estimator.Estimator(
    image_uri           = ECR_URI,
    role                = ROLE,
    instance_count      = 1,
    instance_type       = "ml.m5.large",
    base_job_name       = "chronos-training-job",
    environment         = {
        "TRAINING_DATA_PATH": TRAINING_DATA_PATH,
        "TRAINING_LIMIT_TIME": TRAINING_LIMIT_TIME,
        "BASE_MODEL_PATH": BASE_MODEL_PATH,
        "TUNED_MODEL_PATH": TUNED_MODEL_PATH
    },
    sagemaker_session   = session,
)

estimator.fit()


