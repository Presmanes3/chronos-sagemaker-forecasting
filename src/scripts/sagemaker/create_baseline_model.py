"""
Create a baseline AutoGluon model WITHOUT fine-tuning.

This script wraps a pretrained Chronos model in AutoGluon's TimeSeriesPredictor
structure WITHOUT modifying the weights. This allows:
- Using the same inference code for baseline and fine-tuned models
- Fair comparison (baseline uses original weights)
- Consistent API interface

Usage:
    python src/scripts/sagemaker/create_baseline_model.py
"""

import os
import sys
import boto3
import tarfile
import shutil
from pathlib import Path
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.config import Config
from dotenv import load_dotenv

load_dotenv()

import pandas as pd
from autogluon.timeseries import TimeSeriesPredictor, TimeSeriesDataFrame

# ----- Load configuration
config = Config("./config.yaml")

def list_s3_models(bucket: str, prefix: str) -> list:
    """List all .tar.gz files in the specified S3 bucket/prefix."""
    boto3_session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))
    s3 = boto3_session.client('s3')
    try:
        response = s3.list_objects_v2(Bucket=bucket, Prefix=prefix)
        if 'Contents' not in response:
            return []
        models = [obj['Key'] for obj in response['Contents'] if obj['Key'].endswith('.tar.gz')]
        return models
    except Exception as e:
        print(f"Error listing S3 models: {e}")
        return []

def select_model_interactively(bucket: str, prefix: str) -> str:
    """Prompt user to select a base model from S3."""
    models = list_s3_models(bucket, prefix)
    
    if not models:
        print(f"\n⚠️  No models found in s3://{bucket}/{prefix}")
        sys.exit(1)
    
    print("\n📦 Available base models in S3:\n")
    for i, model in enumerate(models, 1):
        print(f"  {i}. {model}")
    print()
    
    while True:
        try:
            choice = input(f"Select a model (1-{len(models)}): ").strip()
            choice_idx = int(choice) - 1
            
            if 0 <= choice_idx < len(models):
                return f"s3://{bucket}/{models[choice_idx]}"
            else:
                print("Invalid selection. Try again.")
        except (ValueError, KeyError):
            print("Invalid input. Please enter a number.")

def create_minimal_training_data():
    """Create minimal dummy data just to initialize the predictor."""
    data = pd.DataFrame({
        'item_id': ['dummy'] * 48,
        'timestamp': pd.date_range('2024-01-01', periods=48, freq='H'),
        'value': [100.0] * 48  # Constant values
    })
    
    return TimeSeriesDataFrame.from_data_frame(
        data,
        id_column='item_id',
        timestamp_column='timestamp'
    )

def download_s3_model(s3_uri: str, local_path: Path) -> Path:
    """Download base model from S3 and extract it."""
    boto3_session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))
    s3 = boto3_session.client('s3')
    
    # Parse S3 URI
    s3_uri = s3_uri.replace("s3://", "")
    bucket, key = s3_uri.split("/", 1)
    
    # Download tar.gz
    local_tar = local_path / "base_model.tar.gz"
    local_tar.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📥 Downloading from S3: s3://{bucket}/{key}")
    s3.download_file(bucket, key, str(local_tar))
    
    # Extract
    extract_dir = local_path / "base_model"
    extract_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📦 Extracting model...")
    with tarfile.open(local_tar, "r:gz") as tar:
        tar.extractall(extract_dir)
    
    # Find the actual model directory (handle nested structures)
    for item in extract_dir.rglob("config.json"):
        if item.parent.name.startswith("chronos"):
            print(f"✅ Model extracted to: {item.parent}")
            return item.parent
    
    # If not found in subdirectories, might be at root
    if (extract_dir / "config.json").exists():
        return extract_dir
    
    raise FileNotFoundError(f"Could not find Chronos model in extracted archive")

def create_baseline_model(base_model_s3_uri: str, output_name: str):
    """
    Create a baseline model WITHOUT training.
    
    Args:
        base_model_s3_uri: S3 URI of base Chronos model
        output_name: Name for output tar.gz file
    """
    
    print("\n" + "="*80)
    print("🔧 CREATING BASELINE MODEL (NO TRAINING)")
    print("="*80)
    
    # Temporary directories
    work_dir = Path("/tmp/baseline_creation")
    work_dir.mkdir(exist_ok=True, parents=True)
    
    output_dir = work_dir / "model_output"
    output_dir.mkdir(exist_ok=True)
    
    # Download and extract base model
    base_model_path = download_s3_model(base_model_s3_uri, work_dir)
    
    print(f"\n📁 Output directory: {output_dir}")
    
    # Create minimal training data
    print("\n📊 Creating minimal initialization data...")
    dummy_data = create_minimal_training_data()
    
    # Initialize predictor
    print(f"\n🤖 Initializing predictor with base model...")
    predictor = TimeSeriesPredictor(
        prediction_length=24,
        path=str(output_dir),
        target="value",
        eval_metric="RMSE",
    )
    
    # "Train" with time_limit=60 → Enough to initialize model, but not actually train
    print("\n⚡ Initializing predictor (time_limit=60 seconds - minimal training)...")
    print("   Note: This only initializes the model structure, doesn't modify weights\n")
    try:
        predictor.fit(
            train_data=dummy_data,
            time_limit=60,  # Enough time to initialize Chronos model
            hyperparameters={
                "Chronos": {
                    "model_path": str(base_model_path),
                }
            },
            skip_model_selection=True,
        )
    except Exception as e:
        # If it fails, check if at least predictor.pkl was created
        print(f"⚠️  Fit may have issues: {type(e).__name__}: {e}")
        if not (output_dir / "predictor.pkl").exists():
            print("❌ predictor.pkl not created - cannot proceed")
            sys.exit(1)
        print("✅ predictor.pkl exists - continuing anyway")
    
    # Verify structure
    print("\n✅ Verifying model structure...")
    required_files = ["predictor.pkl"]
    for f in required_files:
        if (output_dir / f).exists():
            print(f"  ✅ {f}")
        else:
            print(f"  ❌ {f} - WARNING: Missing file")
    
    # Create tar.gz
    print(f"\n📦 Creating {output_name}...")
    local_tar_path = work_dir / output_name
    
    with tarfile.open(str(local_tar_path), "w:gz") as tar:
        for item in output_dir.iterdir():
            tar.add(str(item), arcname=item.name)
            print(f"  ✅ Added: {item.name}")
    
    tar_size_mb = local_tar_path.stat().st_size / (1024**2)
    print(f"\n✅ Baseline model created: {local_tar_path} ({tar_size_mb:.2f} MB)")
    
    # Upload to S3
    s3_bucket = config["s3"]["bucket"]
    s3_key = f"{config['s3']['production_models']['s3_prefix']}{output_name}"
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    
    print(f"\n⬆️  Uploading to S3: {s3_uri}")
    
    boto3_session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE"))
    s3 = boto3_session.client("s3")
    
    try:
        s3.upload_file(str(local_tar_path), s3_bucket, s3_key)
        print(f"✅ Upload successful!")
        print(f"\n🎯 Baseline model ready for deployment:")
        print(f"   S3 URI: {s3_uri}")
        print(f"   Model name: {output_name}")
        print(f"\n💡 To deploy this baseline:")
        print(f"   python src/scripts/sagemaker/launch_endpoint.py")
        print(f"   Then select: {output_name}")
        
    except Exception as e:
        print(f"❌ Upload failed: {e}")
        sys.exit(1)
    
    # Cleanup (best effort - Windows may lock files)
    print("\n🧹 Cleaning up temporary files...")
    try:
        # Give Windows time to release file handles
        import time
        time.sleep(0.5)
        shutil.rmtree(work_dir)
        print("✅ Cleanup successful")
    except (PermissionError, OSError) as e:
        print(f"⚠️  Could not remove temp files (will be cleaned later): {e}")
        print(f"   Temp directory: {work_dir}")
    
    print("\n" + "="*80)
    print("✅ BASELINE MODEL CREATION COMPLETE")
    print("="*80)
    
    return s3_uri


if __name__ == "__main__":
    # Get S3 configuration
    S3_BUCKET = config["s3"]["bucket"]
    S3_MODELS_PREFIX = config["s3"]["upload"]["models"]["s3_prefix"]
    
    # Select base model interactively
    print("\n" + "="*80)
    print("📦 SELECT BASE MODEL FOR BASELINE")
    print("="*80)
    BASE_MODEL_S3_URI = select_model_interactively(S3_BUCKET, S3_MODELS_PREFIX)
    
    # Prompt for output model name
    print("\n" + "="*80)
    print("📝 OUTPUT BASELINE MODEL CONFIGURATION")
    print("="*80)
    
    default_name = f"baseline-chronos-{datetime.now().strftime('%Y%m%d-%H%M%S')}.tar.gz"
    print(f"\nDefault baseline name: {default_name}")
    print(f"S3 destination: s3://{S3_BUCKET}/{config['s3']['production_models']['s3_prefix']}")
    print()
    
    custom_name = input(f"Enter baseline model name (press Enter for default): ").strip()
    if not custom_name:
        output_model_name = default_name
    else:
        # Ensure .tar.gz extension
        if not custom_name.endswith('.tar.gz'):
            custom_name += '.tar.gz'
        output_model_name = custom_name
    
    print(f"\n✅ Output baseline will be saved as: {output_model_name}")
    print("="*80 + "\n")
    
    # Create baseline model
    create_baseline_model(
        base_model_s3_uri=BASE_MODEL_S3_URI,
        output_name=output_model_name
    )
