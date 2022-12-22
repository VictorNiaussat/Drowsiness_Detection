import logging
import pprint
from environment import create_trainer_environment
import os


log = logging.getLogger("xception")
env = create_trainer_environment()


class Config(object):
    sagemaker_directory_structure = {
        "train_data": "/opt/ml/input/data/training",
        "output_artifacts": "/opt/ml/output",
    }

    def __init__(self, directory_structure = sagemaker_directory_structure):

        
        self.output_artifacts_directory = directory_structure["output_artifacts"]
        self.training_directory = directory_structure["train_data"] 
        self.data_name = os.getenv("data_name")
        self.model_params = self.parse_parameters()
    def parse_parameters(self):
        """Parse the ENV variables [ set in the dockerfile ]
        to determine configuration settings"""

        log.info("parsing model hyperparameters from command line arguments...log")  
        model_params = {
            "input_size": env.hyperparameters.get('input_size', object_type=int, default=299),
            "nb_epochs": env.hyperparameters.get('nb_epochs', object_type=int, default=10),
            "nb_couches_rentrainement": env.hyperparameters.get('nb_couches_rentrainement', object_type=int, default=4),
            "batch_size": env.hyperparameters.get('batch_size', object_type=int, default=4),
        }
        log.info(pprint.pformat(model_params, indent=5))

        return model_params