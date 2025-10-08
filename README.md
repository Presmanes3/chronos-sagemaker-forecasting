# 🕒 Time Series Forecasting with Amazon SageMaker & Hugging Face Chronos

This project demonstrates how to build a **minimal yet production-ready pipeline** for **time series forecasting** using **Amazon SageMaker** and the [Chronos-bolt-tiny](https://huggingface.co/amazon/chronos-bolt-tiny) model from **Hugging Face**.

The original repository of Chronos can be found [here](https://github.com/amazon-science/chronos-forecasting/tree/main).

## 🎯 Project Overview

The goal of this project is to **fine-tune**, **deploy**, and **serve** a Hugging Face model for time series prediction using **SageMaker training and inference infrastructure**, and finally interact with it via a **Streamlit web interface**.

The workflow demonstrates the complete lifecycle of an ML model:
1. Data preparation and upload to S3
2. Model fine-tuning on SageMaker
3. Model deployment as an API endpoint
4. Local Docker-based inference (optional)
5. Frontend interaction with Streamlit

## 🧠 Model

We use **[`amazon/chronos-bolt-tiny`](https://huggingface.co/amazon/chronos-bolt-tiny)** — a lightweight, transformer-based model specialized in **time series forecasting**.

Chronos models are capable of:
- Multivariate forecasting
- Handling missing values
- Capturing temporal dependencies
- Fast inference even on CPU instances

## AWS Set-Up

To execute this project, it is necessary to configure AWS credentials. To do this, first, create a profile:

``$ aws configure --profile <profile-name>``

Here, you will be prompted for the Access Key ID, Secret Access Key, region, and output format. The scripts will use the AWS_PROFILE environment variable to select the profile to use when running the scripts.

It will also be necessary to create an IAM role with the required permissions to execute SageMaker. In this case, a role named ``SageMakerExecutionRole`` has been created with the permissions ``AmazonSageMakerFullAccess`` and ``AmazonS3FullAccess``.

```bash
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```

## ECR Public Login

To create the Docker image with the Docker Compose document, it is first necessary to log in to the public ECR repository.

``aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com``

The response should be ``Login Succeeded``.

If this is the case, you can now build the Docker image and push it to ECR.