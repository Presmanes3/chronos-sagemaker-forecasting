import boto3
import subprocess
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

AWS_PROFILE = os.getenv("AWS_PROFILE")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")
REPO_NAME = os.getenv("ECR_REPO_NAME", "chronos-custom")
IMAGE_TAG = os.getenv("IMAGE_TAG", "latest")
DOCKERFILE_PATH = "deployment/Dockerfile"
DOCKER_CONTEXT = "./deployment"

session = boto3.Session(profile_name=AWS_PROFILE, region_name=AWS_REGION)
ecr_client = session.client("ecr")
sts_client = session.client("sts")

account_id = sts_client.get_caller_identity()["Account"]
repo_uri = f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/{REPO_NAME}:{IMAGE_TAG}"

print(f"📦 AWS Account: {account_id}")
print(f"🧱 Target ECR URI: {repo_uri}")

try:
    ecr_client.describe_repositories(repositoryNames=[REPO_NAME])
    print(f"✅ ECR repository '{REPO_NAME}' already exists.")
except ClientError as e:
    if e.response["Error"]["Code"] == "RepositoryNotFoundException":
        print(f"🆕 Creating ECR repository: {REPO_NAME}")
        ecr_client.create_repository(repositoryName=REPO_NAME)
    else:
        raise e

print("🔐 Logging in to ECR...")
login_pw = ecr_client.get_authorization_token()["authorizationData"][0]["authorizationToken"]
proxy_endpoint = ecr_client.get_authorization_token()["authorizationData"][0]["proxyEndpoint"]

login_cmd = f"aws ecr get-login-password --region {AWS_REGION} --profile {AWS_PROFILE} | docker login --username AWS --password-stdin {account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com"
subprocess.run(login_cmd, shell=True, check=True)
print("✅ Docker authenticated with ECR.")

print("🏗️ Building Docker image...")
subprocess.run(
    ["docker", "build", "-t", f"{REPO_NAME}:{IMAGE_TAG}", "-f", DOCKERFILE_PATH, DOCKER_CONTEXT],
    check=True
)
print("✅ Docker build complete.")

print(f"🏷️ Tagging image for ECR: {repo_uri}")
subprocess.run(
    ["docker", "tag", f"{REPO_NAME}:{IMAGE_TAG}", repo_uri],
    check=True
)

print("☁️ Pushing image to ECR...")
subprocess.run(
    ["docker", "push", repo_uri],
    check=True
)
print(f"🚀 Successfully pushed to ECR: {repo_uri}")
