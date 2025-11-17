import os
import sys
import boto3
import tarfile
import pandas as pd
import tempfile
import shutil
from pathlib import Path
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# =============================================================================
# Load environment variables
# =============================================================================
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("⚠️  python-dotenv not installed — proceeding without .env file.")

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
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
    raise ValueError(f"❌ Missing required environment variables: {', '.join(missing)}")

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
    print(f"⬇️  Downloading {s3_uri} → {local_path}")
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
            print(f"✅ Base Chronos model found at: {p}")
            return str(p)

    sys.exit("❌ No valid Chronos base model found inside archive.")


# =============================================================================
# Step 1 — Prepare model and data
# =============================================================================
session = create_boto3_session(AWS_PROFILE)

# Download and extract the base model
base_model_local = (
    extract_model_from_tar(download_from_s3(BASE_MODEL_PATH, session))
    if BASE_MODEL_PATH.startswith("s3://")
    else BASE_MODEL_PATH
)
print(f"✅ Base model ready at: {base_model_local}")

# Download the training data
training_data_local = (
    download_from_s3(TRAINING_DATA_PATH, session)
    if TRAINING_DATA_PATH.startswith("s3://")
    else TRAINING_DATA_PATH
)
print(f"✅ Training data ready at: {training_data_local}")

# =============================================================================
# Step 2 — Load training dataset
# =============================================================================
df = pd.read_csv(training_data_local)
df["item_id"] = "Turbine_1"
df.rename(columns={"Unnamed: 0": "timestamp"}, inplace=True)
df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_localize(None)

ts_df = TimeSeriesDataFrame.from_data_frame(
    df[["timestamp", "ActivePower", "item_id"]],
    id_column="item_id",
    timestamp_column="timestamp",
)

print(f"📊 Training dataset loaded: {len(ts_df)} time steps | columns = {list(ts_df.columns)}")

# =============================================================================
# Step 3 — Fine-tune model
# =============================================================================
output_dir = "/opt/ml/model/fine_tuned"
print(f"🏗️  Starting fine-tuning → {output_dir}")

predictor = TimeSeriesPredictor(
    prediction_length=24,
    path=output_dir,
    target="ActivePower",
    eval_metric="RMSE",
)

try:
    predictor.fit(
        train_data=ts_df,
        time_limit=TRAINING_LIMIT_TIME,
        hyperparameters={
            "Chronos": {
                "pretrained_model_name": "chronos-bolt-tiny",
                "model_path": base_model_local,
                "save_model_pipeline": True,  # ✅ required for SageMaker inference
            }
        },
    )
    print("✅ Fine-tuning completed successfully.")
except Exception as e:
    print(f"❌ Fine-tuning failed: {e}")
    sys.exit(1)

# =============================================================================
# Step 3.1 — Embed base model artifacts into fine-tuned model
# =============================================================================
base_model_dir = Path("/opt/ml/model/base_model/chronos-bolt-tiny")
target_dir = Path(output_dir) / "models"

for chronos_dir in target_dir.rglob("Chronos*"):
    embedded_dir = chronos_dir / "base_model_artifacts"
    embedded_dir.mkdir(exist_ok=True)
    print(f"📦 Embedding base model artifacts into: {embedded_dir}")

    for fname in ["config.json", "model.safetensors", "tokenizer.json"]:
        src = base_model_dir / fname
        if src.exists():
            shutil.copy(src, embedded_dir / fname)
            print(f"✅ Copied {fname}")
        else:
            print(f"⚠️ Missing {fname}, skipped.")

# =============================================================================
# Step 4 — Verify saved artifacts
# =============================================================================
chronos_dir = Path(output_dir)
print(f"🔍 Verifying model artifacts inside {chronos_dir} ...")

expected_files = ["predictor.pkl", "learner.pkl"]
for f in expected_files:
    if not (chronos_dir / f).exists():
        print(f"⚠️ Warning: expected {f} not found in fine-tuned directory.")

if list(chronos_dir.rglob("config.json")):
    print("✅ Chronos configuration files detected.")
else:
    print("⚠️ Warning: No Chronos configuration files found. Inference may fail.")

# =============================================================================
# Step 5 — Finalize for SageMaker export
# =============================================================================
print("\n========================================================")
print("📦 FINALIZING MODEL EXPORT")
print("SageMaker will now automatically package everything under:")
print("   /opt/ml/model/")
print("and upload it to the designated S3 output path.")
print("--------------------------------------------------------")

if not Path("/opt/ml/model/base_model").exists():
    print("⚠️ Base model directory missing under /opt/ml/model/")
if not Path("/opt/ml/model/fine_tuned").exists():
    print("⚠️ Fine-tuned model directory missing under /opt/ml/model/")

print("✅ All steps completed successfully.")
print("========================================================\n")

# =============================================================================
# Step 6 — Compress full /opt/ml/model and upload to TUNED_MODEL_PATH
# =============================================================================


print("\n========================================================")
print("📦 Creating final model.tar.gz for manual upload...")
print("========================================================")

model_root = "/opt/ml/model"
local_tar_path = "/tmp/model.tar.gz"

# Create tar.gz of full /opt/ml/model
with tarfile.open(local_tar_path, "w:gz") as tar:
    tar.add(model_root, arcname=".")
    print(f"✅ Created archive at: {local_tar_path}")

# Parse tuned model S3 path
if not TUNED_MODEL_PATH.startswith("s3://"):
    raise ValueError("TUNED_MODEL_PATH must be an S3 URI, e.g. s3://bucket/path/model.tar.gz")

bucket = TUNED_MODEL_PATH.replace("s3://", "").split("/", 1)[0]
key = TUNED_MODEL_PATH.replace(f"s3://{bucket}/", "")

print(f"⬆️  Uploading model.tar.gz → {TUNED_MODEL_PATH}")

s3 = session.client("s3")
s3.upload_file(local_tar_path, bucket, key)

print("✅ Model uploaded successfully!")
print("========================================================\n")