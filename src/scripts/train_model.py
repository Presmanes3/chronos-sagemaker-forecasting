import os
import sys
import boto3
import tempfile
import tarfile
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# ----------------------------------------------------------------------------- 
# Load environment
# ----------------------------------------------------------------------------- 
load_dotenv()

# ----------------------------------------------------------------------------- 
# Configuration parameters
# ----------------------------------------------------------------------------- 
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH")
TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH")
S3_BUCKET_NAME = os.getenv("AWS_S3_BUCKET")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")
S3_SUBFOLDER = "fine-tunned"
OUTPUT_DIR = "fine_tuned_model"

# ----------------------------------------------------------------------------- 
# Validation of environment
# ----------------------------------------------------------------------------- 
if not BASE_MODEL_PATH:
    sys.exit("❌ BASE_MODEL_PATH environment variable is not set.")
if not TRAINING_DATA_PATH:
    sys.exit("❌ TRAINING_DATA_PATH environment variable is not set.")
if not S3_BUCKET_NAME:
    sys.exit("❌ AWS_S3_BUCKET environment variable is not set.")

print(f"Base model path: {BASE_MODEL_PATH}")
print(f"Training data path: {TRAINING_DATA_PATH}")

# ----------------------------------------------------------------------------- 
# Helper functions
# ----------------------------------------------------------------------------- 
def download_from_s3(s3_uri: str, profile: str = None) -> str:
    """Downloads a file from S3 and returns its local path."""
    assert s3_uri.startswith("s3://"), "Invalid S3 URI"
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")

    _, _, bucket_and_key = s3_uri.partition("s3://")
    bucket, key = bucket_and_key.split("/", 1)
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))

    print(f"Downloading from S3: s3://{bucket}/{key}")
    s3.download_file(bucket, key, local_path)
    print(f"Download complete: {local_path}")
    return local_path


def get_local_model_path(base_model_path: str, profile: str = None) -> str:
    """Ensures the model directory is local. Downloads + extracts if needed."""
    if not base_model_path.startswith("s3://"):
        print("Using local model directory.")
        return base_model_path

    tmp_dir = tempfile.mkdtemp(prefix="chronos_base_")
    extract_dir = Path(tmp_dir) / "model_extracted"

    local_tar = download_from_s3(base_model_path, profile)
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    # Find the folder containing Chronos files
    possible_dirs = [extract_dir] + [p for p in extract_dir.rglob("*") if p.is_dir()]
    for p in possible_dirs:
        if (p / "config.json").exists() and (p / "model.safetensors").exists():
            print(f"Using Chronos model directory: {p}")
            return str(p)

    sys.exit(f"No valid Chronos model found in extracted path: {extract_dir}")


def get_local_data_path(data_path: str, profile: str = None) -> str:
    """Ensures the dataset is available locally (downloads if from S3)."""
    if not data_path.startswith("s3://"):
        print("Using local training dataset.")
        return data_path

    local_data = download_from_s3(data_path, profile)
    return local_data


def compress_model_folder(folder_path: str) -> str:
    """Compresses a folder into a temporary .tar.gz and returns its path."""
    archive_path = os.path.join(tempfile.gettempdir(), "fine_tuned_chronos_model.tar.gz")
    with tarfile.open(archive_path, "w:gz") as tar:
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            tar.add(file_path, arcname=file_name)
    return archive_path


# ----------------------------------------------------------------------------- 
# Step 1: Prepare model and data
# ----------------------------------------------------------------------------- 
BASE_MODEL_LOCAL_PATH = get_local_model_path(BASE_MODEL_PATH, profile=AWS_PROFILE)
TRAINING_DATA_LOCAL_PATH = get_local_data_path(TRAINING_DATA_PATH, profile=AWS_PROFILE)

# ----------------------------------------------------------------------------- 
# Step 2: Load and preprocess training data
# ----------------------------------------------------------------------------- 
df = pd.read_csv(TRAINING_DATA_LOCAL_PATH)
df["item_id"] = "Turbine_1"
df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)
clean_df = df[["timestamp", "ActivePower", "item_id"]]

ts_df = TimeSeriesDataFrame.from_data_frame(
    clean_df,
    id_column="item_id",
    timestamp_column="timestamp"
)

# ----------------------------------------------------------------------------- 
# Step 3: Train model with error handling
# ----------------------------------------------------------------------------- 
try:
    predictor = TimeSeriesPredictor(
        prediction_length=24,
        path=OUTPUT_DIR,
        target="ActivePower",
        eval_metric="RMSE"
    )

    print("🚀 Fine-tuning Chronos model...")
    predictor.fit(
        train_data=ts_df,
        time_limit=10,
        hyperparameters={
            "Chronos": {
                "pretrained_model_name": "chronos_bolt_tiny",
                "model_path": BASE_MODEL_LOCAL_PATH
            }
        }
    )

    print("Fine-tuning complete!")
    predictor.save()
    print(f"Fine-tuned model saved at: {OUTPUT_DIR}")

except Exception as e:
    print(f"Training failed: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------------- 
# Step 4: Compress fine-tuned model
# ----------------------------------------------------------------------------- 
try:
    archive_path = compress_model_folder(OUTPUT_DIR)
    print(f"Model compressed at: {archive_path}")
except Exception as e:
    print(f"Failed to compress model: {e}")
    sys.exit(1)

# ----------------------------------------------------------------------------- 
# Step 5: Upload fine-tuned model to S3
# ----------------------------------------------------------------------------- 
try:
    print(f"Uploading fine-tuned model to s3://{S3_BUCKET_NAME}/{S3_SUBFOLDER}/ ...")
    session = boto3.Session(profile_name=AWS_PROFILE)
    s3 = session.client("s3")
    s3_key = f"{S3_SUBFOLDER}/fine_tuned_chronos_model.tar.gz"
    s3.upload_file(archive_path, S3_BUCKET_NAME, s3_key)
    print("Fine-tuned model uploaded successfully!")
    print(f"S3 path: s3://{S3_BUCKET_NAME}/{s3_key}")

except Exception as e:
    print(f"Failed to upload to S3: {e}")
    sys.exit(1)
