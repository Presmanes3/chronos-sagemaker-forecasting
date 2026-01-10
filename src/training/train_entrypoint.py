import os
import sys
import boto3
import tarfile
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame


from dotenv import load_dotenv
load_dotenv()


# ----- Load environment variables
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH")
TRAINING_DATA_PATH = os.getenv("TRAINING_DATA_PATH")
TUNED_MODEL_PATH = os.getenv("TUNED_MODEL_PATH")  # keeping env var name for compatibility
AWS_PROFILE = os.getenv("AWS_PROFILE")
TRAINING_LIMIT_TIME = int(os.getenv("TRAINING_LIMIT_TIME", "100"))

if not BASE_MODEL_PATH or not TRAINING_DATA_PATH or not TUNED_MODEL_PATH:
    missing = [
        k for k, v in {
            "BASE_MODEL_PATH": BASE_MODEL_PATH,
            "TRAINING_DATA_PATH": TRAINING_DATA_PATH,
            "TUNNED_MODEL_PATH": TUNED_MODEL_PATH,
        }.items() if v is None
    ]
    raise ValueError(f"Missing required environment variables: {', '.join(missing)}")

print(f"""
================== ENVIRONMENT CONFIG ==================
BASE_MODEL_PATH     = {BASE_MODEL_PATH}
TRAINING_DATA_PATH  = {TRAINING_DATA_PATH}
TUNED_MODEL_PATH    = {TUNED_MODEL_PATH}
AWS_PROFILE         = {AWS_PROFILE}
TRAINING_LIMIT_TIME = {TRAINING_LIMIT_TIME}s
========================================================
""")

# =============================================================================
# Helper functions
# =============================================================================
def create_boto3_session(profile=None):
    """Create a boto3 session compatible with SageMaker or local runs."""
    if os.getenv("SM_TRAINING_ENV"):  # inside SageMaker
        return boto3.Session()
    return boto3.Session(profile_name=profile) if profile else boto3.Session()


def download_from_s3(s3_uri: str, session) -> str:
    """Download an S3 object and return its local path."""
    assert s3_uri.startswith("s3://"), f"Invalid S3 URI: {s3_uri}"
    bucket, key = s3_uri.replace("s3://", "").split("/", 1)
    local_path = os.path.join(tempfile.gettempdir(), os.path.basename(key))
    s3 = session.client("s3")
    print(f"Downloading {s3_uri} -> {local_path}")
    s3.download_file(bucket, key, local_path)
    return local_path


def extract_model_from_tar(tar_path: str) -> str:
    """Extract a tar.gz model archive into /opt/ml/model/base_model and return path to Chronos model folder."""
    extract_dir = Path("/opt/ml/model/base_model")
    extract_dir.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=extract_dir)

    for p in extract_dir.rglob("*"):
        if (p / "config.json").exists() and (p / "model.safetensors").exists():
            print(f"Base Chronos model found at: {p}")
            return str(p)

    sys.exit("No valid Chronos base model found inside archive.")


# ----- Create sesion
session = create_boto3_session(AWS_PROFILE)

# Download and extract the base model
base_model_local = (
    extract_model_from_tar(download_from_s3(BASE_MODEL_PATH, session))
    if BASE_MODEL_PATH.startswith("s3://")
    else BASE_MODEL_PATH
)
print(f"Base model ready at: {base_model_local}")

# Download the training data
training_data_local = (
    download_from_s3(TRAINING_DATA_PATH, session)
    if TRAINING_DATA_PATH.startswith("s3://")
    else TRAINING_DATA_PATH
)
print(f"Training data ready at: {training_data_local}")

# ----- Load training data into TimeSeriesDataFrame
df = pd.read_csv(training_data_local)
df["item_id"] = "Turbine_1"
df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

ts_df = TimeSeriesDataFrame.from_data_frame(
    df[["timestamp", "ActivePower", "item_id"]],
    id_column="item_id",
    timestamp_column="timestamp",
)

print(f"Training dataset loaded: {len(ts_df)} time steps | columns = {list(ts_df.columns)}")

# ----- Fine-tune the model
output_dir = "/opt/ml/model" 
print(f"Starting fine-tuning -> {output_dir}")

# Configuration aligned with generate_dataset.py
PREDICTION_LENGTH = 48  
CONTEXT_LENGTH = 336    

predictor = TimeSeriesPredictor(
    prediction_length=PREDICTION_LENGTH,
    path=output_dir,
    target="ActivePower",
    eval_metric="RMSE",
    freq="30T",  
)

try:
    predictor.fit(
        train_data=ts_df,
        time_limit=TRAINING_LIMIT_TIME,
        hyperparameters={
            "Chronos": {
                "pretrained_model_name": "chronos-bolt-tiny",
                "model_path": base_model_local,
                "save_model_pipeline": True,  
                "context_length": CONTEXT_LENGTH,
                "fine_tune": True,
                "fine_tune_steps": 2000,  # Number of training steps
                "fine_tune_lr": 5e-5,     # Conservative learning rate
            }
        },
        num_val_windows=5,  
        verbosity=2,
    )
    print("Fine-tuning completed successfully.")
except Exception as e:
    print(f"Fine-tuning failed: {e}")
    sys.exit(1)

# ----- Embed base model artifacts into fine-tuned model directories
# Use the actual base_model_local path that was already verified to exist
base_model_source = Path(base_model_local)
target_dir = Path(output_dir) / "models"

print(f"\nEmbedding base model artifacts from: {base_model_source}")

# Define all possible base model files to copy
base_model_files = [
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "generation_config.json",
    "tokenizer_config.json",
]

for chronos_dir in target_dir.rglob("Chronos*"):
    if not chronos_dir.is_dir():
        continue
        
    embedded_dir = chronos_dir / "base_model_artifacts"
    embedded_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nTarget directory: {embedded_dir}")

    copied_count = 0
    for fname in base_model_files:
        src = base_model_source / fname
        if src.exists():
            dest = embedded_dir / fname
            shutil.copy2(src, dest)
            print(f"   Copied: {fname} ({src.stat().st_size / 1024:.1f} KB)")
            copied_count += 1
        else:
            print(f"   Missing: {fname}")
    
    print(f"\nEmbedded {copied_count}/{len(base_model_files)} base model files")

# ----- Verify fine-tuned model artifacts
chronos_dir = Path(output_dir)
print(f"Verifying model artifacts inside {chronos_dir} ...")

expected_files = ["predictor.pkl", "learner.pkl"]
for f in expected_files:
    if not (chronos_dir / f).exists():
        print(f"Warning: expected {f} not found in fine-tuned directory.")

if list(chronos_dir.rglob("config.json")):
    print("Chronos configuration files detected.")
else:
    print("Warning: No Chronos configuration files found. Inference may fail.")

# ----- Verify model structure
print("\n========================================================")
print("VERIFYING MODEL STRUCTURE")
print("========================================================")

model_dir = Path(output_dir)
print(f"\nModel directory: {model_dir}")
print(f"\nContents:")
for item in model_dir.iterdir():
    if item.is_file():
        print(f"  [FILE] {item.name} ({item.stat().st_size / 1024:.1f} KB)")
    else:
        print(f"  [DIR]  {item.name}/")

print("\nModel training completed successfully.")
print("========================================================\n")

# ----- Create clean tar.gz with proper structure
print("\n========================================================")
print("Creating model.tar.gz with clean structure...")
print("========================================================

local_tar_path = "/tmp/model.tar.gz"

# Create tar.gz with clean structure (files at root)
# IMPORTANT: Include base_model/ directory - AutoGluon references it during inference
with tarfile.open(local_tar_path, "w:gz") as tar:
    for item in Path(output_dir).iterdir():
        tar.add(str(item), arcname=item.name)
        print(f"Added to archive: {item.name}")

print(f"Archive created at: {local_tar_path}")

# Parse tuned model S3 path
if not TUNED_MODEL_PATH.startswith("s3://"):
    raise ValueError("TUNED_MODEL_PATH must be an S3 URI, e.g. s3://bucket/path/model.tar.gz")

bucket = TUNED_MODEL_PATH.replace("s3://", "").split("/", 1)[0]
key = TUNED_MODEL_PATH.replace(f"s3://{bucket}/", "")

print(f"Uploading model.tar.gz -> {TUNED_MODEL_PATH}")

s3 = session.client("s3")
s3.upload_file(local_tar_path, bucket, key)

print("Model uploaded successfully!")
print("========================================================\n")