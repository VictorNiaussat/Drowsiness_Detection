import boto3
sm_client = boto3.client('sagemaker')

import json
from pyexpat import model
from time import strftime, localtime

date = strftime('%Y-%m-%d-%H-%M-%S', localtime())


batch_size = "32"
nb_epochs = "20"
input_size="299"
nb_epochs="10"
nb_couches_rentrainement="4"


role_arn = "arn:aws:iam::266875515584:role/sagemaker-execution-role"

data_name = "state-farm-distracted-driver-detection.zip"
data_s3_url = f"s3://drowsiness-detection-bucket/data-kaggle/{data_name}"
training_job_name = f"xception-training-{date}"
training_image_uri = "266875515584.dkr.ecr.eu-west-1.amazonaws.com/training-xception:latest"
training_instance_type =  "ml.m4.xlarge"

model_name = f'xception-model-{date}'
model_data_url = f"s3://drowsiness-detection-bucket/models/{training_job_name}/output/model.tar.gz"
inference_image_url = training_image_uri


config_endpoint_name = f'xception-inference-endpoint-config-{date}'
config_endpoint_instance_type = "ml.m4.xlarge"


if __name__=='__main__':
    response = sm_client.create_training_job(
                                TrainingJobName=training_job_name,
                                HyperParameters={
                                    'batch_size': batch_size,
                                    'nb_epochs': nb_epochs,
                                    'input_size': input_size,
                                    'nb_couches_rentrainement': nb_couches_rentrainement,
                                },
                                AlgorithmSpecification={
                                    'TrainingImage': training_image_uri,
                                    'TrainingInputMode': 'File',
                                },
                                RoleArn=role_arn,
                                InputDataConfig=[
                                    {
                                        'ChannelName': 'training',
                                        'DataSource': {
                                            'S3DataSource': {
                                                'S3DataType': 'S3Prefix',
                                                'S3Uri': data_s3_url,

                                            },
                                        },
                                        'CompressionType': 'None',
                                        'InputMode': 'File',
                                    },
                                ],
                                OutputDataConfig={
                                    'S3OutputPath': 's3://drowsiness-detection-bucket/models/'
                                },
                                ResourceConfig={
                                    'InstanceType': training_instance_type,
                                    'InstanceCount': 1,
                                    "VolumeSizeInGB":25,
                                    
                                },
                                StoppingCondition={
                                                    'MaxRuntimeInSeconds': 36000
                                                },
                                TensorBoardOutputConfig={
                                    'S3OutputPath': 'https://drowsiness-detection-bucket.s3.eu-west-1.amazonaws.com/logs/'
                                },
                                Environment={
                                                'data_name': data_name
                                            }
                            )
    
    print(response)
    """
    response = sm_client.create_model(
                        ModelName=model_name,
                        PrimaryContainer={
                            'ContainerHostname': 'string',
                            'Image': model_data_url,
                            'Mode': 'SingleModel',
                            'ModelDataUrl': model_data_url,
                            'Environment': {
                                    'input_size': input_size,
                            },
                        },
                        ExecutionRoleArn=role_arn,
                    )
    
    print(response)
    
    response = sm_client.create_endpoint_config(
                                EndpointConfigName=config_endpoint_name,
                                ProductionVariants=[
                                    {
                                        'VariantName': 'model1',
                                        'ModelName': model_name,
                                        'InitialInstanceCount': 1,
                                        'InstanceType': config_endpoint_instance_type,
                                    },]
                            )
    
    print(response)"""