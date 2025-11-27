import os
import boto3
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.config import Config

# ----- Load configuration
config = Config("./config.yaml")

ENDPOINT_NAME = config["sagemaker"]["endpoint_name"]


aws_profile = os.getenv("AWS_PROFILE")
if not aws_profile:
    raise ValueError("❌ AWS_PROFILE environment variable not set")

# Initialize sessions
boto_session = boto3.Session(profile_name=aws_profile)
sagemaker_client = boto_session.client("sagemaker")

def delete_sagemaker_endpoint_and_config(endpoint_name: str):
    
    endpoint_config_name = endpoint_name + "-config" # Assuming config name matches endpoint name
    endpoint_model_name  = endpoint_name + "-model"  # Assuming model name matches endpoint name

    try:
        print(f"Deleting SageMaker endpoint: {endpoint_name}")
        sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
        print(f"Endpoint '{endpoint_name}' deleted successfully.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint '{endpoint_name}' not found.")
        else:
            raise

    try:
        print(f"Deleting endpoint configuration: {endpoint_config_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
        print(f"Endpoint configuration '{endpoint_config_name}' deleted successfully.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint config '{endpoint_config_name}' not found.")
        else:
            raise

    try:
        print(f"Deleting model: {endpoint_model_name}")
        sagemaker_client.delete_model(ModelName=endpoint_model_name)
        print(f"Model '{endpoint_model_name}' deleted successfully.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Model '{endpoint_model_name}' not found.")
        else:
            raise


if __name__ == "__main__":
    
    delete_sagemaker_endpoint_and_config(ENDPOINT_NAME)
