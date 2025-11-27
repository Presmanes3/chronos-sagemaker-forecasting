import os
import boto3
import sys

from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from src.config import Config

# ----- Load configuration
config = Config("./config.yaml")

# Retrieve the S3 bucket name from environment variables
DATASET_BASE = config["paths"]["dataset_base"]

if not DATASET_BASE or not DATASET_BASE.startswith('s3://'):
    raise ValueError("DATASET_BASE must be set and start with 's3://'")

# Extract bucket name from DATASET_BASE (s3://bucket-name/optional/prefix)
s3_parts = DATASET_BASE.replace('s3://', '').split('/', 1)
AWS_S3_BUCKET = s3_parts[0]
s3_prefix = s3_parts[1] if len(s3_parts) > 1 else ''

# Ask user for file path and S3 key
file_path = input("Enter the file path to upload: ").strip()
file_key = input("Enter the S3 destination path (e.g., /test/data.csv): ").strip()

# Initialize S3 client
s3 = boto3.client('s3')

try:
    # Combine prefix with user input, removing leading slash
    full_key = os.path.join(s3_prefix, file_key.lstrip('/')).replace('\\', '/')
    
    # Upload the file to S3
    s3.upload_file(file_path, AWS_S3_BUCKET, full_key)
    print(f"File '{file_path}' successfully uploaded to bucket '{AWS_S3_BUCKET}' as '{full_key}'.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except NoCredentialsError:
    print("Error: AWS credentials not found.")
except Exception as e:
    print(f"An error occurred: {e}")