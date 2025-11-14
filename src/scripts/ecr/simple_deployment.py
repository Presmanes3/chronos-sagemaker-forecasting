from sagemaker.model import Model
from sagemaker import get_execution_role

from sagemaker import image_uris

image = image_uris.retrieve(
    framework="autogluon",
    region="eu-west-1",
    image_scope="inference",
    version="1.3",
    instance_type="ml.m5.large"  # 👈 necesario
)

print(image)

role = "arn:aws:iam::296062581113:role/SageMakerExecutionRole"

model = Model(
    image_uri   = image,
    model_data  = "s3://chronos-forecasting-presmanes/fine-tunned/fine_tuned_chronos_model.tar.gz",
    role        = role
)

predictor = model.deploy(
    initial_instance_count  = 1,
    instance_type           = "ml.m5.large",
    endpoint_name           = "chronos-forecasting-endpoint"
)
