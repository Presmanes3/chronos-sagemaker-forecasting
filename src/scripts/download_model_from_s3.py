import os
import boto3
from pathlib import Path
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Load environment variables
# -----------------------------------------------------------------------------
load_dotenv()
DEFAULT_BUCKET = os.getenv("AWS_S3_BUCKET", "")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")

# Default local path to store downloaded models
LOCAL_MODELS_DIR = Path("./models")
LOCAL_MODELS_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def list_models_in_s3(bucket: str, prefix: str = "models/", profile: str = None):
    """List all .tar.gz model artifacts available in S3 under a prefix."""
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")

    print(f"🔍 Listing models from s3://{bucket}/{prefix} ...")

    response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
    if "Contents" not in response:
        print("⚠️ No models found in this bucket/prefix.")
        return []

    models = [
        obj["Key"]
        for obj in response["Contents"]
        if obj["Key"].endswith(".tar.gz")
    ]
    return models


def download_from_s3(bucket: str, key: str, dest_dir: Path, profile: str = None):
    """Download a .tar.gz model from S3 to the local models directory."""
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")

    file_name = os.path.basename(key)
    dest_path = dest_dir / file_name

    print(f"⬇️  Downloading s3://{bucket}/{key} → {dest_path}")
    s3.download_file(bucket, key, str(dest_path))
    print(f"✅ Download complete: {dest_path}")
    return dest_path


# -----------------------------------------------------------------------------
# Main CLI logic
# -----------------------------------------------------------------------------
def main():
    print("📥 Chronos Model Downloader")
    print("============================\n")

    # Step 1: Ask user for bucket (default from .env)
    bucket = input(f"Enter S3 bucket name [{DEFAULT_BUCKET}]: ").strip() or DEFAULT_BUCKET
    if not bucket:
        print("❌ No bucket provided and no AWS_S3_BUCKET in .env.")
        return

    # Step 2: List models available
    models = list_models_in_s3(bucket=bucket, prefix="models/", profile=AWS_PROFILE)
    if not models:
        return

    print("\nAvailable models in S3:\n")
    for i, key in enumerate(models, 1):
        print(f"  {i}. {os.path.basename(key)}")

    print()
    choice = input("Enter the number(s) of the model(s) to download (comma-separated, e.g. 1,2): ")
    selected_indices = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
    selected_models = [models[i - 1] for i in selected_indices if 1 <= i <= len(models)]

    if not selected_models:
        print("⚠️ No valid selection made.")
        return

    # Step 3: Download selected models
    for key in selected_models:
        download_from_s3(bucket=bucket, key=key, dest_dir=LOCAL_MODELS_DIR, profile=AWS_PROFILE)

    print("\n🎉 All selected models have been downloaded successfully!")
    print(f"🗂️  Files saved in: {LOCAL_MODELS_DIR.resolve()}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
