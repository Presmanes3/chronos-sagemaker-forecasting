# Time Series Forecasting with Amazon SageMaker & Chronos

Production-ready pipeline for time series forecasting using Amazon SageMaker and [Chronos-bolt-tiny](https://huggingface.co/amazon/chronos-bolt-tiny).

Original Chronos repository: [amazon-science/chronos-forecasting](https://github.com/amazon-science/chronos-forecasting)

## Overview

End-to-end ML pipeline for time series prediction:

1. Data preparation and upload to S3
2. Model fine-tuning on SageMaker (AutoGluon + Chronos)
3. Model deployment as REST API endpoint
4. Local Docker-based inference (FastAPI + Uvicorn)
5. Model comparison and evaluation

## Model

[`amazon/chronos-bolt-tiny`](https://huggingface.co/amazon/chronos-bolt-tiny) - Lightweight transformer model for time series forecasting.

Features:
- Multivariate forecasting
- Missing value handling
- Temporal dependency capture
- Fast CPU inference

## Project Structure

```
src/
├── config.py                    # Configuration loader
├── deployment/                  # Inference service
│   ├── serve.py                 # FastAPI application
│   ├── predictor_service.py     # AutoGluon predictor wrapper
│   ├── models.py                # Pydantic request/response models
│   ├── utils.py                 # Data parsing utilities
│   └── dockerfile               # Deployment container
├── training/                    # Training job
│   ├── train_entrypoint.py      # Fine-tuning script
│   └── dockerfile               # Training container
└── scripts/
    ├── compare_models.py        # Model comparison tool
    ├── dataset/                 # Data processing
    │   └── generate_dataset.py  # Dataset preparation
    ├── ecr/                      # Container registry
    │   ├── push_deployment_image.py
    │   └── push_training_image.py
    ├── s3/                       # S3 operations
    │   ├── upload_data_to_s3.py
    │   └── upload_base_model_to_s3.py
    └── sagemaker/               # SageMaker operations
        ├── launch_training_job.py
        ├── launch_endpoint.py
        └── destroy_endpoint.py

test/
├── unit/                        # Unit tests
└── e2e/                         # End-to-end tests
    └── test_inference.py
```

## Configuration

All paths and settings are managed in `config.yaml`:

```yaml
s3:
  bucket: chronos-presmanes

sagemaker:
  training:
    limit_time: 300
    instance_type: ml.g4dn.xlarge
  inference:
    instance_type: ml.t2.medium
  endpoint_name: chronos-endpoint-prod
```

## Usage

### 1. Prepare Dataset

```bash
python src/scripts/dataset/generate_dataset.py
```

### 2. Upload to S3

```bash
python src/scripts/s3/upload_data_to_s3.py
```

### 3. Build and Push Docker Images

```bash
python src/scripts/ecr/push_training_image.py
python src/scripts/ecr/push_deployment_image.py
```

### 4. Launch Training Job

```bash
python src/scripts/sagemaker/launch_training_job.py
```

### 5. Deploy Endpoint

```bash
python src/scripts/sagemaker/launch_endpoint.py
```

### 6. Test Inference

```bash
python test/e2e/test_inference.py
```

### 7. Compare Models

```bash
python src/scripts/compare_models.py --test-data data/wind-power-forecasting/processed/split/Turbine_data_processed_test.csv
```

## Architecture

![Architecture Diagram](docs/images/diagram.svg)

## AWS Setup

### Configure Profile

```bash
aws configure --profile <profile-name>
```

### Create IAM Role

```bash
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

### ECR Login

```bash
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com
```

## Environment Variables

Required in `.env`:

```
AWS_PROFILE=<your-profile>
AWS_REGION=eu-west-1
AWS_SAGEMAKER_ROLE_ARN=<role-arn>
AWS_ECR_TRAINING_IMAGE_URI=<training-image-uri>
AWS_ECR_DEPLOYMENT_IMAGE_URI=<deployment-image-uri>
```

## Local Development

Run inference locally:

```bash
cd src/deployment
python serve.py
```

Or with Docker:

```bash
docker-compose up
```

## TODO

- [x] EDA for testing base model locally
- [x] Training script with AutoGluon locally
- [x] Training job in SageMaker
- [x] Deploy model to SageMaker endpoint
- [x] Inference endpoint testing
- [ ] Streamlit app for user interaction