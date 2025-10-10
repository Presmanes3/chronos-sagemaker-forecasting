import os
import boto3
from botocore.exceptions import NoCredentialsError
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the S3 bucket name from environment variables
AWS_S3_BUCKET = os.getenv('AWS_S3_BUCKET')

if not AWS_S3_BUCKET:
    raise ValueError("AWS_S3_BUCKET environment variable is not set.")

# File to upload
file_path = './data/wind-power-forecasting/Turbine_Data.csv'
file_key = "data/" + os.path.basename(file_path)  # Use the file name as the S3 object key

# Initialize S3 client
s3 = boto3.client('s3')

try:
    # Upload the file to S3
    s3.upload_file(file_path, AWS_S3_BUCKET, file_key)
    print(f"File '{file_path}' successfully uploaded to bucket '{AWS_S3_BUCKET}' as '{file_key}'.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except NoCredentialsError:
    print("Error: AWS credentials not found.")
except Exception as e:
    print(f"An error occurred: {e}")