from collections import namedtuple

DataIngestionArtifact = namedtuple("DataIngestionArtifact",[
    "train_file_path",
    "test_file_path",  
    "is_ingested",
    "message"])

DataValidationArtifact = namedtuple("DataValidationArtifact",
["schema_file_path","is_validated","message","validated_train_path","validated_test_path"])



DataTransformationArtifact = namedtuple("DataTransformationArtifact",["is_transformed",
                                                                    "message",
                                                                    "transformed_train_file_path",
                                                                    "transformed_test_file_path",
                                                                    "preprocessed_object_file_path",
                                                        "feature_engineering_object_file_path"])



ModelTrainerArtifact =namedtuple("ModelTrainerArtifact",[
                                                            "is_trained",
                                                            "message",
                                                            "trained_model_file_path",
                                                            "model_artifact_report",
                                                            "train_mse",
                                                            "test_mse",
                                                            "train_r2",
                                                            "test_r2",
                                                            "model_r2"
                                                        ])


ModelEvaluationArtifact=namedtuple("ModelEvaluationArtifact",["model_name",
                                                              "R2_score",
                                                              "selected_model_path",
                                                              "model_report_path"])

ModelPusherArtifact=namedtuple("ModelPusherArtifact",["message"])
