import os
import boto3
import sagemaker

from dotenv import load_dotenv

load_dotenv()



boto3_session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))

session = sagemaker.Session(boto_session=boto3_session)

ecr_uri = os.environ.get("AWS_ECR_TRAINING_IMAGE_URI")

if ecr_uri is None:
    raise ValueError("AWS_ECR_TRAINING_IMAGE_URI environment variable is not set.")

role = os.environ.get("AWS_SAGEMAKER_ROLE_ARN")
if role is None:
    raise ValueError("AWS_SAGEMAKER_ROLE_ARN environment variable is not set.")

s3_bucket = os.environ.get("AWS_S3_BUCKET")
if s3_bucket is None:
    raise ValueError("AWS_S3_BUCKET environment variable is not set.")

base_model_path = os.environ.get("BASE_MODEL_PATH")
if base_model_path is None:
    raise ValueError("BASE_MODEL_PATH environment variable is not set.")

training_data_path = os.environ.get("TRAINING_DATA_PATH")
if training_data_path is None:
    raise ValueError("TRAINING_DATA_PATH environment variable is not set.")

training_time_limit = os.environ.get("TRAINING_LIMIT_TIME", "3600")

estimator = sagemaker.estimator.Estimator(
    image_uri           = ecr_uri,
    role                = role,
    instance_count      = 1,
    instance_type       = "ml.m5.large",
    base_job_name       = "chronos-training-job",
    environment         = {
        "AWS_S3_BUCKET": s3_bucket,
        "TRAINING_DATA_PATH": training_data_path,
        "TRAINING_LIMIT_TIME": training_time_limit,
        "BASE_MODEL_PATH": base_model_path,
        "S3_SUBFOLDER" : "fine-tunned",
        "OUTPUT_DIR": "models/fine-tunned/"
    },
    sagemaker_session   = session,
)

estimator.fit()
