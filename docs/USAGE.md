# Usage Guide

Step-by-step guide to train and deploy the forecasting model.

## 1. Prepare Dataset

```bash
python src/scripts/dataset/generate_dataset.py
```

## 2. Upload to S3

```bash
python src/scripts/s3/upload_data_to_s3.py
```

## 3. Build and Push Docker Images

```bash
python src/scripts/ecr/push_training_image.py
python src/scripts/ecr/push_deployment_image.py
```

## 4. Launch Training Job

```bash
python src/scripts/sagemaker/launch_training_job.py
```

## 5. Deploy Endpoint

```bash
python src/scripts/sagemaker/launch_endpoint.py
```

## 6. Test Inference

```bash
python test/e2e/test_inference.py
```

## 7. Compare Models

```bash
python src/scripts/compare_models.py --test-data data/wind-power-forecasting/processed/split/Turbine_data_processed_test.csv
```

## 8. Destroy Endpoint (when done)

```bash
python src/scripts/sagemaker/destroy_endpoint.py
```
