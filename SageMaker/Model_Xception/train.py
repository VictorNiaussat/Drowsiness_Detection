#Necessary Imports
from config import Config
import logging
import sys
import traceback
from Training import training_model
from preprocessing import preprocessing_data, unzip_data

def train():

    config_training = Config()
    unzip_data(config_training)
    nb_classes, traingen, valgen = preprocessing_data(config_training)
    training_model(config_training, nb_classes, traingen, valgen)
    


def configure_logging():
    hpo_log = logging.getLogger("xception")
    log_handler = logging.StreamHandler()
    log_handler.setFormatter(
        logging.Formatter("%(asctime)-15s %(levelname)8s %(name)s %(message)s")
    )
    hpo_log.addHandler(log_handler)
    hpo_log.setLevel(logging.DEBUG)
    hpo_log.propagate = False


if __name__ == "__main__":
    configure_logging()
    try:
        train()
        sys.exit(0)  # success exit code
    except Exception:
        traceback.print_exc()
        sys.exit(-1)  # failure exit code

