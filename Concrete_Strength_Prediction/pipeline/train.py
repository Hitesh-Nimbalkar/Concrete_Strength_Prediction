from collections import namedtuple
from datetime import datetime
import uuid
from Concrete_Strength_Prediction.configuration import *
from Concrete_Strength_Prediction.logger import logging
from Concrete_Strength_Prediction.exception import ApplicationException
from threading import Thread
from typing import List
from Concrete_Strength_Prediction.utils.utils import read_yaml_file
from multiprocessing import Process
from Concrete_Strength_Prediction.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from Concrete_Strength_Prediction.components.data_ingestion import DataIngestion
from Concrete_Strength_Prediction.components.data_validation import DataValidation




import os, sys
from collections import namedtuple
from datetime import datetime
import pandas as pd



class Pipeline():

    def __init__(self, config: Configuration = Configuration()) -> None:
        try:
            self.config = config
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            data_ingestion = DataIngestion(data_ingestion_config=self.config.get_data_ingestion_config())
            return data_ingestion.initiate_data_ingestion()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
        
    def start_data_validation(self, data_ingestion_artifact:DataIngestionArtifact)-> DataValidationArtifact:
        try:
            data_validation = DataValidation(data_validation_config=self.config.get_data_validation_config(),
                                             data_ingestion_artifact=data_ingestion_artifact)
            return data_validation.initiate_data_validation()
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    def run_pipeline(self):
        try:
             #data ingestion

            data_ingestion_artifact = self.start_data_ingestion()
            data_validation_artifact=self.start_data_validation(data_ingestion_artifact=data_ingestion_artifact)


         
        except Exception as e:
            raise ApplicationException(e, sys) from e