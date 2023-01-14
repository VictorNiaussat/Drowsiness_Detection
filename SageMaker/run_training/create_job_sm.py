import sagemaker
from time import strftime, localtime
date = strftime('%Y-%m-%d-%H-%M-%S', localtime())
import boto3

batch_size = "32"
nb_epochs = "20"
input_size="299"
nb_epochs="20"
nb_couches_rentrainement="4"



role_arn = "arn:aws:iam::266875515584:role/sagemaker-execution-role" #Droit sagemaker s3

data_name = "state-farm-distracted-driver-detection.zip"
data_s3_url = f"s3://drowsiness-detection-bucket/data-kaggle/{data_name}" #lien 
training_job_name = f"xception-training-{date}"
training_image_uri = "266875515584.dkr.ecr.eu-west-1.amazonaws.com/training-xception:latest" #lien vers docker : 
training_instance_type =  "ml.m4.xlarge" #instance entrainement

model_name = f'xception-model-{date}'
model_data_url = f"s3://drowsiness-detection-bucket/models/{training_job_name}/output/model.tar.gz" #endroit où stock poids
inference_image_url = training_image_uri


config_endpoint_name = f'xception-inference-endpoint-config-{date}' #endpoint : serveur où model est conserver -- message avec flask
config_endpoint_instance_type = "ml.m4.xlarge"


if __name__=='__main__':
    sm_client = boto3.client('sagemaker')
    sess = sagemaker.Session(boto_session=boto3.Session(region_name='eu-west-1'))
    xception =  sagemaker.estimator.Estimator(
            training_image_uri,
            role_arn,
            instance_count=1,
            instance_type=training_instance_type,
            output_path='s3://drowsiness-detection-bucket/models/',
            input_mode= "File",
            sagemaker_session=sess,
            hyperparameters={
                            'batch_size': batch_size,
                            'nb_epochs': nb_epochs,
                            'input_size': input_size,
                            'nb_couches_rentrainement': nb_couches_rentrainement,
                                    },
            environment={
                        'data_name': data_name
                    }
        )

    xception.fit(inputs=data_s3_url, wait=True, logs='All', job_name=training_job_name)

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
    
    print(response)