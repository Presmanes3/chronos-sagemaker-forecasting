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


BASE_MODEL_PATH     = config["paths"]["base_model"] # S3 path to base model which will be fine-tuned

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


