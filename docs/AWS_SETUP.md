# AWS Setup

Configuration guide for AWS services.

## Prerequisites

- AWS CLI installed
- Docker installed
- AWS account with SageMaker access

## Configure AWS Profile

```bash
aws configure --profile <profile-name>
```

You will be prompted for:
- Access Key ID
- Secret Access Key
- Default region (e.g., `eu-west-1`)
- Output format (e.g., `json`)

## Create IAM Role

Create a SageMaker execution role with required permissions.

### 1. Create the role

```bash
aws iam create-role \
  --role-name SageMakerExecutionRole \
  --assume-role-policy-document file://trust-policy.json
```

### 2. Attach SageMaker policy

```bash
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
```

### 3. Attach S3 policy

```bash
aws iam attach-role-policy \
  --role-name SageMakerExecutionRole \
  --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### 4. Get role ARN

```bash
aws iam get-role --role-name SageMakerExecutionRole --query 'Role.Arn' --output text
```

## ECR Login

Login to AWS public ECR for base images:

```bash
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

Login to your private ECR:

```bash
aws ecr get-login-password --region <region> --profile <profile> | \
  docker login --username AWS --password-stdin <account-id>.dkr.ecr.<region>.amazonaws.com
```

## Environment Variables

Create a `.env` file in the project root:

```bash
AWS_PROFILE=<your-profile>
AWS_REGION=eu-west-1
AWS_SAGEMAKER_ROLE_ARN=arn:aws:iam::<account-id>:role/SageMakerExecutionRole
AWS_ECR_TRAINING_IMAGE_URI=<account-id>.dkr.ecr.<region>.amazonaws.com/chronos-training:latest
AWS_ECR_DEPLOYMENT_IMAGE_URI=<account-id>.dkr.ecr.<region>.amazonaws.com/chronos-deployment:latest
AWS_ECR_TRAINING_REPO_NAME=chronos-training
AWS_ECR_DEPLOYMENT_REPO_NAME=chronos-deployment
```

## S3 Bucket Setup

Create the S3 bucket for model artifacts and data:

```bash
aws s3 mb s3://chronos-presmanes --region eu-west-1
```

Create folder structure:

```bash
aws s3api put-object --bucket chronos-presmanes --key models/base/
aws s3api put-object --bucket chronos-presmanes --key models/production/
aws s3api put-object --bucket chronos-presmanes --key data/
```

## Verify Setup

Check AWS configuration:

```bash
aws sts get-caller-identity --profile <profile-name>
```

List S3 buckets:

```bash
aws s3 ls --profile <profile-name>
```

List ECR repositories:

```bash
aws ecr describe-repositories --profile <profile-name>
```
