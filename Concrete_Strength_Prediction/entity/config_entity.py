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

DataTransformationConfig = namedtuple("DataTransformationConfig",["transformed_train_dir",
                                                                  "transformed_test_dir",
                                                                  "preprocessed_object_file_path",
                                                                  "feature_engineering_object_file_path"])
ModelTrainerConfig = namedtuple("ModelTrainerConfig",["trained_model_file_path","model_config_path","report_path"])



saved_model_config = SavedModelConfig(saved_model_file_path=saved_model_file_path,
                                            saved_report_file_path=saved_report_file_path)
                                    