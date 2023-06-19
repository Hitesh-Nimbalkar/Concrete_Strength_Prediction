
import os
from datetime import datetime

def get_current_time_stamp():
    return f"{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"

CURRENT_TIME_STAMP = get_current_time_stamp()

ROOT_DIR = os.getcwd()  #to get current working directory
CONFIG_DIR = "config"
CONFIG_FILE_NAME = "config.yaml"
CONFIG_FILE_PATH = os.path.join(ROOT_DIR,CONFIG_DIR,CONFIG_FILE_NAME)

FILE_NAME='ConcreteStrengthData.csv'


from Concrete_Strength_Prediction.constant.database import *

from Concrete_Strength_Prediction.constant.training_pipeline import *
from Concrete_Strength_Prediction.constant.training_pipeline.data_ingestion import *

