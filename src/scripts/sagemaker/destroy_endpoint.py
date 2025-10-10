import os
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

ENDPOINT_NAME = os.getenv("AWS_SAGEMAKER_ENDPOINT_NAME")

aws_profile = os.getenv("AWS_PROFILE")
if not aws_profile:
    raise ValueError("❌ AWS_PROFILE environment variable not set")

# Initialize sessions
boto_session = boto3.Session(profile_name=aws_profile)
sagemaker_client = boto_session.client("sagemaker")

def delete_sagemaker_endpoint_and_config(endpoint_name: str):

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
        print(f"Deleting endpoint configuration: {endpoint_name}")
        sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print(f"Endpoint configuration '{endpoint_name}' deleted successfully.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Endpoint config '{endpoint_name}' not found.")
        else:
            raise

    try:
        print(f"Deleting model: {endpoint_name}")
        sagemaker_client.delete_model(ModelName=endpoint_name)
        print(f"Model '{endpoint_name}' deleted successfully.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            print(f"Model '{endpoint_name}' not found.")
        else:
            raise


if __name__ == "__main__":
    
    delete_sagemaker_endpoint_and_config(ENDPOINT_NAME)
