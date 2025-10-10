import os
import boto3
import sagemaker

from dotenv import load_dotenv


load_dotenv()



boto3_session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))

session = sagemaker.Session(boto_session=boto3_session)

BASE_MODEL_PATH     = os.getenv("BASE_MODEL_PATH")
TRAINING_DATA_PATH  = os.getenv("TRAINING_DATA_PATH")
TUNNED_MODEL_PATH   = os.getenv("TUNNED_MODEL_PATH")
AWS_PROFILE         = ""
TRAINING_LIMIT_TIME = os.getenv("TRAINING_LIMIT_TIME", "3600")

ECR_URI             = os.getenv("AWS_ECR_TRAINING_IMAGE_URI")
ROLE                = os.getenv("AWS_SAGEMAKER_ROLE_ARN")

if not BASE_MODEL_PATH or not TRAINING_DATA_PATH or not TUNNED_MODEL_PATH or not ECR_URI or not ROLE:
    missing = [
        k for k, v in {
            "BASE_MODEL_PATH": BASE_MODEL_PATH,
            "TRAINING_DATA_PATH": TRAINING_DATA_PATH,
            "TUNNED_MODEL_PATH": TUNNED_MODEL_PATH,
            "AWS_ECR_TRAINING_IMAGE_URI": ECR_URI,
            "AWS_SAGEMAKER_ROLE_ARN": ROLE,
        }.items() if v is None
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

print(f"""System Variables:
      - BASE_MODEL_PATH:     {BASE_MODEL_PATH}
      - TRAINING_DATA_PATH:  {TRAINING_DATA_PATH}
      - TUNNED_MODEL_PATH:   {TUNNED_MODEL_PATH}
      - AWS_PROFILE:         {AWS_PROFILE}
      - TRAINING_LIMIT_TIME: {TRAINING_LIMIT_TIME} seconds
      - ECR_URI:             {ECR_URI}
      - ROLE:                {ROLE}
      """)

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
        "TUNNED_MODEL_PATH": TUNNED_MODEL_PATH,
        "AWS_PROFILE": AWS_PROFILE,
    },
    sagemaker_session   = session,
)

estimator.fit()
