import boto3
import subprocess
import os
from botocore.exceptions import ClientError
from dotenv import load_dotenv


def load_env_variables():
    """Load environment variables for training ECR push."""
    load_dotenv()
    return {
        "AWS_PROFILE": os.getenv("AWS_PROFILE"),
        "AWS_REGION": os.getenv("AWS_REGION", "eu-west-1"),
        "REPO_NAME": os.getenv("ECR_TRAINING_REPO_NAME", "chronos-training"),
        "IMAGE_TAG": os.getenv("IMAGE_TAG", "latest"),
        "DOCKERFILE_PATH": "./src/training/dockerfile",
        "DOCKER_CONTEXT": "./src/training"
    }


def get_aws_clients(profile, region):
    """Initialize AWS clients."""
    session = boto3.Session(profile_name=profile, region_name=region)
    return session.client("ecr"), session.client("sts")


def get_account_id(sts_client):
    """Retrieve AWS account ID."""
    return sts_client.get_caller_identity()["Account"]


def ensure_ecr_repository(ecr_client, repo_name):
    """Ensure the ECR repository exists."""
    try:
        ecr_client.describe_repositories(repositoryNames=[repo_name])
        print(f"✅ ECR repository '{repo_name}' already exists.")
    except ClientError as e:
        if e.response["Error"]["Code"] == "RepositoryNotFoundException":
            print(f"🆕 Creating ECR repository: {repo_name}")
            ecr_client.create_repository(repositoryName=repo_name)
        else:
            raise e


def docker_login(account_id, region, profile):
    """Log in to ECR using Docker."""
    print("🔐 Logging in to ECR...")
    login_cmd = (
        f"aws ecr get-login-password --region {region} --profile {profile} | "
        f"docker login --username AWS --password-stdin {account_id}.dkr.ecr.{region}.amazonaws.com"
    )
    subprocess.run(login_cmd, shell=True, check=True)
    print("✅ Docker authenticated with ECR.")


def build_and_push_docker_image(repo_uri, repo_name, image_tag, dockerfile_path, docker_context):
    """Build, tag, and push Docker image."""
    print("🏗️ Building training Docker image...")
    subprocess.run(
        ["docker", "build", "-t", f"{repo_name}:{image_tag}", "-f", dockerfile_path, docker_context],
        check=True
    )
    print("✅ Docker build complete.")

    print(f"🏷️ Tagging image for ECR: {repo_uri}")
    subprocess.run(
        ["docker", "tag", f"{repo_name}:{image_tag}", repo_uri],
        check=True
    )

    print("☁️ Pushing image to ECR...")
    subprocess.run(
        ["docker", "push", repo_uri],
        check=True
    )
    print(f"🚀 Successfully pushed training image to ECR: {repo_uri}")


def main():
    """Build and push training Docker image to ECR."""
    env = load_env_variables()

    # Initialize AWS clients
    ecr_client, sts_client = get_aws_clients(env["AWS_PROFILE"], env["AWS_REGION"])

    # Get account ID and repository URI
    account_id = get_account_id(sts_client)
    repo_uri = f"{account_id}.dkr.ecr.{env['AWS_REGION']}.amazonaws.com/{env['REPO_NAME']}:{env['IMAGE_TAG']}"

    print(f"📦 AWS Account: {account_id}")
    print(f"🧱 Target ECR Training URI: {repo_uri}")

    # Ensure ECR repository exists
    ensure_ecr_repository(ecr_client, env["REPO_NAME"])

    # Authenticate Docker with ECR
    docker_login(account_id, env["AWS_REGION"], env["AWS_PROFILE"])

    # Build, tag, and push image
    build_and_push_docker_image(
        repo_uri,
        env["REPO_NAME"],
        env["IMAGE_TAG"],
        env["DOCKERFILE_PATH"],
        env["DOCKER_CONTEXT"]
    )

    print("\n✅ Training image successfully pushed to ECR.")
    print(f"   URI: {repo_uri}")


if __name__ == "__main__":
    main()
