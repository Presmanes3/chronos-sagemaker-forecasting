# TimeSeriesForecasting

# AWS Set-Up
To execute this project, it is necessary to configure AWS credentials. To do this, first, create a profile:

``$ aws configure --profile <profile-name>``

Here, you will be prompted for the Access Key ID, Secret Access Key, region, and output format. The scripts will use the AWS_PROFILE environment variable to select the profile to use when running the scripts.

It will also be necessary to create an IAM role with the required permissions to execute SageMaker. In this case, a role named ``SageMakerExecutionRole`` has been created with the permissions ``AmazonSageMakerFullAccess`` and ``AmazonS3FullAccess``.

```bash
aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document file://trust-policy.json

aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess

aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
```
