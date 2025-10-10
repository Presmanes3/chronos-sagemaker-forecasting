import os
import tarfile
import boto3
from dotenv import load_dotenv

load_dotenv()
DEFAULT_BUCKET = os.getenv("AWS_S3_BUCKET", "")
AWS_PROFILE = os.getenv("AWS_PROFILE", "default")

MODELS_DIR = "./models"

def list_model_folders(models_dir: str):
    return [f for f in os.listdir(models_dir) if os.path.isdir(os.path.join(models_dir, f))]

def compress_folder(folder_path: str, output_path: str):
    """Compress the selected folder into a .tar.gz archive."""
    with tarfile.open(output_path, "w:gz") as tar:
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    return output_path

def upload_to_s3(file_path: str, bucket: str, s3_key: str, profile: str):
    """Upload a file to S3 using the specified AWS profile."""
    session = boto3.Session(profile_name=profile)
    s3 = session.client("s3")
    print(f"Uploading {file_path} → s3://{bucket}/{s3_key}")
    s3.upload_file(file_path, bucket, s3_key)
    print("✅ Upload complete!")

def main():
    print("📦 Chronos Model Uploader")
    print("==========================\n")

    if not os.path.exists(MODELS_DIR):
        print(f"❌ Folder '{MODELS_DIR}' does not exist.")
        return

    model_folders = list_model_folders(MODELS_DIR)
    if not model_folders:
        print("⚠️ No model folders found inside ./models/")
        return

    print("Available models:\n")
    for i, folder in enumerate(model_folders, 1):
        print(f"  {i}. {folder}")
    print()

    choice = input("Enter the number(s) of the model(s) to upload (comma-separated, e.g. 1,3): ")
    selected_indices = [int(x.strip()) for x in choice.split(",") if x.strip().isdigit()]
    selected_folders = [model_folders[i - 1] for i in selected_indices if 1 <= i <= len(model_folders)]

    if not selected_folders:
        print("⚠️ No valid selection made.")
        return

    bucket = input(f"Enter S3 bucket name [{DEFAULT_BUCKET}]: ").strip() or DEFAULT_BUCKET
    if not bucket:
        print("❌ No bucket provided and no AWS_S3_BUCKET in .env.")
        return

    for folder in selected_folders:
        folder_path = os.path.join(MODELS_DIR, folder)
        
        archive_path = os.path.join(MODELS_DIR, f"{folder}.tar.gz")

        print(f"\n🗜️ Compressing '{folder}'...")
        compress_folder(folder_path, archive_path)
        print(f"✅ Created archive: {archive_path}")

        s3_key = f"models/base/{os.path.basename(archive_path)}"
        upload_to_s3(archive_path, bucket, s3_key, AWS_PROFILE)

    print("\n🎉 All selected models have been uploaded successfully!")

if __name__ == "__main__":
    main()
