from collections import namedtuple

#Training - Artifact
TrainingPipelineConfig = namedtuple("TrainingPipelineConfig", ["artifact_dir"])

# Data Ingestion
DataIngestionConfig=namedtuple("DataIngestionConfig",[
    "raw_data_dir",
    "ingested_train_dir",
    "ingested_test_dir"
    ])


DataValidationConfig = namedtuple("DataValidationConfig", ["schema_file_path","validated_train_path","validated_test_path"])

