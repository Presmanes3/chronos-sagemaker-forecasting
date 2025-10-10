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

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------
def list_prefixes(bucket: str, prefix: str, profile: str = None):
    """List all subfolders (prefixes) under a given prefix in an S3 bucket."""
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    result = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter="/")

    prefixes = []
    for page in result:
        if "CommonPrefixes" in page:
            for cp in page["CommonPrefixes"]:
                prefixes.append(cp["Prefix"])
    return prefixes


def list_files_in_prefix(bucket: str, prefix: str, profile: str = None):
    """List all files under a given prefix in S3."""
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")

    paginator = s3.get_paginator("list_objects_v2")
    result = paginator.paginate(Bucket=bucket, Prefix=prefix)

    files = []
    for page in result:
        if "Contents" in page:
            for obj in page["Contents"]:
                if obj["Key"].endswith(".tar.gz"):
                    files.append(obj["Key"])
    return files


def download_from_s3(bucket: str, key: str, dest_dir: Path, profile: str = None):
    """Download a .tar.gz model from S3."""
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")

    file_name = os.path.basename(key)
    dest_dir.mkdir(parents=True, exist_ok=True)
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

    # Step 1: Ask user for bucket
    bucket = input(f"Enter S3 bucket name [{DEFAULT_BUCKET}]: ").strip() or DEFAULT_BUCKET
    if not bucket:
        print("❌ No bucket provided and no AWS_S3_BUCKET in .env.")
        return

    base_prefix = "models/"
    print(f"🔍 Scanning s3://{bucket}/{base_prefix} ...\n")

    # Step 2: List available subfolders (e.g., base/, fine-tunned/)
    prefixes = list_prefixes(bucket, base_prefix, AWS_PROFILE)
    if not prefixes:
        print("⚠️ No model folders found in this bucket.")
        return

    print("Available model folders:\n")
    for i, prefix in enumerate(prefixes, 1):
        print(f"  {i}. {prefix}")

    print()
    folder_choice = input("Select the folder number to explore: ").strip()
    if not folder_choice.isdigit() or not (1 <= int(folder_choice) <= len(prefixes)):
        print("⚠️ Invalid selection.")
        return

    selected_prefix = prefixes[int(folder_choice) - 1]
    print(f"\n📁 Selected folder: s3://{bucket}/{selected_prefix}\n")

    # Step 3: List models inside the selected folder
    models = list_files_in_prefix(bucket, selected_prefix, AWS_PROFILE)
    if not models:
        print("⚠️ No .tar.gz models found inside this folder.")
        return

    print("Available models:\n")
    for i, key in enumerate(models, 1):
        print(f"  {i}. {os.path.basename(key)}")

    print()
    model_choice = input("Enter the number(s) of the model(s) to download (comma-separated, e.g. 1,2): ")
    selected_indices = [int(x.strip()) for x in model_choice.split(",") if x.strip().isdigit()]
    selected_files = [models[i - 1] for i in selected_indices if 1 <= i <= len(models)]

    if not selected_files:
        print("⚠️ No valid selection made.")
        return

    # Step 4: Download selected models
    local_target_dir = Path("./models") / Path(selected_prefix).name.strip("/")
    for key in selected_files:
        download_from_s3(bucket, key, dest_dir=local_target_dir, profile=AWS_PROFILE)

    print("\n🎉 All selected models have been downloaded successfully!")
    print(f"🗂️  Files saved in: {local_target_dir.resolve()}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    main()
