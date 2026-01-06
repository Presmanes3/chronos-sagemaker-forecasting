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

# Retrieve the S3 configuration from config.yaml
AWS_S3_BUCKET = config["s3"]["bucket"]
S3_DATA_PREFIX = config["s3"]["upload"]["data"]["s3_prefix"]

# Ask user for file path and S3 key
file_path = input("Enter the file path to upload: ").strip()
file_key = input(f"Enter the S3 destination path relative to '{S3_DATA_PREFIX}' (e.g., test/data.csv): ").strip()

# Initialize S3 client
s3 = boto3.client('s3')

try:
    # Combine prefix with user input, removing leading slash
    full_key = os.path.join(S3_DATA_PREFIX, file_key.lstrip('/')).replace('\\', '/')
    
    # Upload the file to S3
    s3.upload_file(file_path, AWS_S3_BUCKET, full_key)
    print(f"File '{file_path}' successfully uploaded to bucket '{AWS_S3_BUCKET}' as '{full_key}'.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except NoCredentialsError:
    print("Error: AWS credentials not found.")
except Exception as e:
    print(f"An error occurred: {e}")