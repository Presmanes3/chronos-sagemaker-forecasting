import os
import sys
import boto3
import tempfile
import tarfile
import pandas as pd
from pathlib import Path

from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# ----------------------------------------------------------------------------- 
# Load environment
# ----------------------------------------------------------------------------- 
try :
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("python-dotenv not installed, proceeding without loading .env file")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_MODEL_PATH     = os.getenv("BASE_MODEL_PATH")
TRAINING_DATA_PATH  = os.getenv("TRAINING_DATA_PATH")
TUNNED_MODEL_PATH   = os.getenv("TUNNED_MODEL_PATH")
AWS_PROFILE         = os.getenv("AWS_PROFILE", "default")
TRAINING_LIMIT_TIME = int(os.getenv("TRAINING_LIMIT_TIME", "3600"))

if not BASE_MODEL_PATH or not TRAINING_DATA_PATH or not TUNNED_MODEL_PATH:
    missing = [
        k for k, v in {
            "BASE_MODEL_PATH": BASE_MODEL_PATH,
            "TRAINING_DATA_PATH": TRAINING_DATA_PATH,
            "TUNNED_MODEL_PATH": TUNNED_MODEL_PATH,
        }.items() if v is None
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

print(f"""System Variables:
      - BASE_MODEL_PATH:     {BASE_MODEL_PATH}
      - TRAINING_DATA_PATH:  {TRAINING_DATA_PATH}
      - TUNNED_MODEL_PATH:   {TUNNED_MODEL_PATH}
      - AWS_PROFILE:         {AWS_PROFILE}
      - TRAINING_LIMIT_TIME: {TRAINING_LIMIT_TIME} seconds
      """)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def create_boto3_session(profile=None):
    """Create a boto3 session compatible with SageMaker and local environments."""
    if os.getenv("SM_TRAINING_ENV"):
        return boto3.Session()
    return boto3.Session(profile_name=profile) if profile else boto3.Session()


def download_from_s3(s3_uri: str, session) -> str:
    """Download file from S3 and return local path."""
    assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3 = session.client("s3")
    print(f"⬇️  Downloading {s3_uri} → {local_path}")
    s3.download_file(bucket, key, local_path)
    return local_path


def extract_model_from_tar(tar_path: str) -> str:
    """Extract tar.gz and return directory containing model files."""
    extract_dir = tempfile.mkdtemp(prefix="chronos_model_")
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)
    for p in Path(extract_dir).rglob("*"):
        if (p / "config.json").exists() and (p / "model.safetensors").exists():
            return str(p)
    sys.exit("❌ No valid Chronos model found inside archive.")


def compress_model(folder_path: str) -> str:
    """Compress a folder into a temporary .tar.gz archive."""
    archive_path = os.path.join(tempfile.gettempdir(), "fine_tuned_chronos_model.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        for file in os.listdir(folder_path):
            tar.add(os.path.join(folder_path, file), arcname=file)
    return archive_path


def upload_to_s3(local_path: str, s3_uri: str, session):
    """Upload local file to a specific S3 URI."""
    assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    print(f"⬆️  Uploading {local_path} → {s3_uri}")
    s3 = session.client("s3")
    s3.upload_file(local_path, bucket, key)
    print("✅ Upload complete!")


# -----------------------------------------------------------------------------
# Step 1: Prepare model and data
# -----------------------------------------------------------------------------
session = create_boto3_session(AWS_PROFILE)

base_model_local = (
    extract_model_from_tar(download_from_s3(BASE_MODEL_PATH, session))
    if BASE_MODEL_PATH.startswith("s3://")
    else BASE_MODEL_PATH
)

training_data_local = (
    download_from_s3(TRAINING_DATA_PATH, session)
    if TRAINING_DATA_PATH.startswith("s3://")
    else TRAINING_DATA_PATH
)

# -----------------------------------------------------------------------------
# Step 2: Load training data
# -----------------------------------------------------------------------------
df = pd.read_csv(training_data_local)
df["item_id"] = "Turbine_1"
df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

ts_df = TimeSeriesDataFrame.from_data_frame(
    df[["timestamp", "ActivePower", "item_id"]],
    id_column="item_id",
    timestamp_column="timestamp",
)

# -----------------------------------------------------------------------------
# Step 3: Fine-tune model
# -----------------------------------------------------------------------------
output_dir = tempfile.mkdtemp(prefix="chronos_finetuned_")
print(f"🏗️  Fine-tuning Chronos model → {output_dir}")

predictor = TimeSeriesPredictor(
    prediction_length   = 24,
    path                = output_dir,
    target              = "ActivePower",
    eval_metric         = "RMSE",
)

predictor.fit(
    train_data      = ts_df,
    time_limit      = TRAINING_LIMIT_TIME,
    hyperparameters = {
        "Chronos": {
            "pretrained_model_name": "chronos_bolt_tiny",
            "model_path": base_model_local,
        }
    },
)

# -----------------------------------------------------------------------------
# Step 4: Compress and upload fine-tuned model
# -----------------------------------------------------------------------------
archive_path = compress_model(output_dir)
upload_to_s3(archive_path, TUNNED_MODEL_PATH, session)

print("🎯 Fine-tuning workflow completed successfully!")
print(f"📦 Model uploaded to: {TUNNED_MODEL_PATH}")