import os
import subprocess
import boto3
from botocore.exceptions import ClientError
from dotenv import load_dotenv

def load_env_variables():
    """Load required environment variables."""
    load_dotenv()
    return {
        "AWS_PROFILE": os.getenv("AWS_PROFILE"),
        "AWS_REGION": os.getenv("AWS_REGION", "eu-west-1"),
        "REPO_NAME": os.getenv("ECR_DEPLOYMENT_REPO_NAME", "chronos-deployment"),
        "IMAGE_TAG": os.getenv("IMAGE_TAG", "latest"),
        "DOCKERFILE_PATH": "./src/deployment/Dockerfile",
        "DOCKER_CONTEXT": "./src/deployment"
    }

def get_aws_clients(profile, region):
    """Initialize AWS clients."""
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("ecr"), session.client("sts")

def get_account_id(sts_client):
    """Get AWS account ID."""
    return sts_client.get_caller_identity()["Account"]

def ensure_ecr_repository(ecr_client, repo_name):
    """Ensure target ECR repository exists."""
    try:
        ecr_client.describe_repositories(repositoryNames=[repo_name])
        print(f"✅ ECR repository '{repo_name}' already exists.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            print(f"🆕 Creating ECR repository: {repo_name}")
            ecr_client.create_repository(repositoryName=repo_name)
        else:
            raise e

def docker_login_base_image():
    """Login to AWS SageMaker base image ECR."""
    print("🔐 Logging into SageMaker base image ECR (us-east-1)...")
    subprocess.run(
        "aws ecr get-login-password --region us-east-1 | "
        "docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com",
        shell=True,
        check=True
    )
    print("✅ Authenticated with SageMaker base image ECR.")

def docker_login(account_id, region, profile):
    """Login to user's ECR registry."""
    print("🔐 Logging into user ECR...")
    cmd = (
        f"aws ecr get-login-password --region {region} --profile {profile} | "
        f"docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    )
    subprocess.run(cmd, shell=True, check=True)
    print("✅ Authenticated with user ECR.")

def build_and_push_docker_image(repo_uri, repo_name, image_tag, dockerfile_path, docker_context):
    """Build, tag and push Docker image for SageMaker."""
    print("🏗️ Building Docker image (linux/amd64)...")
    env = os.environ.copy()
    env["DOCKER_BUILDKIT"] = "0"
    subprocess.run([
        "docker",
        "build",
        "--no-cache",
        "--platform=linux/amd64",
        "--provenance=false",
        "--output", "oci-mediatypes=false,type=image",
        "-t", f"{repo_name}:{image_tag}",
        "-f", dockerfile_path,
        docker_context
    ], check=True)
    print("✅ Docker build complete.")

    print(f"🏷️ Tagging image: {repo_uri}")
    subprocess.run(["docker", "tag", f"{repo_name}:{image_tag}", repo_uri], check=True)

    print("☁️ Pushing image to ECR...")
    subprocess.run(["docker", "push", repo_uri], check=True)
    print(f"🚀 Successfully pushed: {repo_uri}")

def main():
    env = load_env_variables()
    ecr_client, sts_client = get_aws_clients(env["AWS_PROFILE"], env["AWS_REGION"])
    account_id = get_account_id(sts_client)
    repo_uri = f"{account_id}.dkr.ecr.{env['AWS_REGION']}.amazonaws.com/{env['REPO_NAME']}:{env['IMAGE_TAG']}"

    print(f"📦 AWS Account: {account_id}")
    print(f"🧱 Target URI:  {repo_uri}")

    ensure_ecr_repository(ecr_client, env["REPO_NAME"])
    # docker_login_base_image()
    docker_login(account_id, env["AWS_REGION"], env["AWS_PROFILE"])
    build_and_push_docker_image(repo_uri, env["REPO_NAME"], env["IMAGE_TAG"], env["DOCKERFILE_PATH"], env["DOCKER_CONTEXT"])

if __name__ == "__main__":
    main()
