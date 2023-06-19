from Concrete_Strength_Prediction.entity.config_entity import DataIngestionConfig
from Concrete_Strength_Prediction.entity.artifact_entity import DataIngestionArtifact
from Concrete_Strength_Prediction.configuration import *
import os,sys
from Concrete_Strength_Prediction.logger import logging
from Concrete_Strength_Prediction.pipeline.train import Pipeline
from Concrete_Strength_Prediction.exception import ApplicationException

def main():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()

    except Exception as e:
        logging.error(f"{e}")
        print(e)

if __name__ == "__main__":
    main()
