import ast
import logging
import sys
import time
import os
import pandas as pd
from sklearn.metrics import make_scorer, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from Concrete_Strength_Prediction.logger import logging
from Concrete_Strength_Prediction.exception import ApplicationException
from Concrete_Strength_Prediction.utils.utils import save_object,read_yaml_file,load_object
from Concrete_Strength_Prediction.entity.config_entity import ModelTrainerConfig,DataValidationConfig
from Concrete_Strength_Prediction.entity.artifact_entity import DataTransformationArtifact
from Concrete_Strength_Prediction.entity.artifact_entity import ModelTrainerArtifact
from Concrete_Strength_Prediction.constant import *
from Concrete_Strength_Prediction.entity.model_factory import evaluate_regression_model
from Concrete_Strength_Prediction.entity.model_factory import *
import sys
import re
from Concrete_Strength_Prediction.entity.model_factory import ModelFactory


class Predictor:
    def __init__(self, preprocessing_object, trained_model_object):
        """
        TrainedModel constructor
        preprocessing_object: preprocessing_object
        trained_model_object: trained_model_object
        """
        self.preprocessing_object = preprocessing_object
        self.trained_model_object = trained_model_object

    def predict(self, X):
        """
        function accepts raw inputs and then transformed raw input using preprocessing_object
        which guarantees that the inputs are in the same format as the training data
        At last it performs prediction on transformed features
        """
        transformed_feature = self.preprocessing_object.transform(X)
        return self.trained_model_object.predict(transformed_feature)







class ModelTrainer:

    def __init__(self, model_trainer_config: ModelTrainerConfig,
                 data_transformation_artifact: DataTransformationArtifact):
        try:
            logging.info(f"{'>>' * 30}Model trainer log started.{'<<' * 30} ")
            self.model_trainer_config = model_trainer_config
            self.data_transformation_artifact = data_transformation_artifact

            
            ## Schema Yaml 
            self.schema_data=read_yaml_file(SCHEMA_FILE_PATH)
            
            self.target_column=self.schema_data[TARGET_COLUMN_KEY]
            
            # Model.yaml path 
            self.model_config_path=self.model_trainer_config.model_config_path
            
            
            
            
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
        
    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Finding transformed Training and Test")
            transformed_train_file_path = self.data_transformation_artifact.transformed_train_file_path
            transformed_test_file_path = self.data_transformation_artifact.transformed_test_file_path

            logging.info("Transformed Data found!!! Now, converting it into dataframe")
            train_df = pd.read_csv(transformed_train_file_path)
            test_df = pd.read_csv(transformed_test_file_path)

            target_column_name = self.target_column

            logging.info("Splitting Input features and Target Feature for train and test data")
            # Train csv
            train_target_feature = train_df[target_column_name]
            train_input_feature = train_df.drop(columns=target_column_name, axis=1)
            
            # Xtrain , Ytrain
            X_train=train_input_feature
            Y_train=train_target_feature
            
            # Test CSV
            test_target_feature = test_df[target_column_name]
            test_input_feature = test_df.drop(columns=target_column_name, axis=1)
            # X_test, Y_test
            X_test=test_input_feature
            Y_test=test_target_feature
            
            # model_factory.py-----> model_trainer.py
            logging.info(f"Initializing model factory class using above model config file: {self.model_config_path}")
            model_factory = ModelFactory(model_config_path=self.model_config_path)
            
            
            logging.info(f"Initiating operation model selection")
            base_r2=0.6
            best_model = model_factory.get_best_model(X=X_train, y=Y_train,base_r2=base_r2)
            
            logging.info(f"-------------")
            
            logging.info(f"Best model found on training dataset: {best_model}")
            
            logging.info(f"-------------")


            logging.info(f"Extracting trained model list.")
            grid_searched_best_model_list: List[GridSearchedBestModel] = model_factory.grid_searched_best_model_list

            model_list = [model.best_model for model in grid_searched_best_model_list]
            logging.info(f"Evaluation all trained model on training and testing dataset both")
            metric_info: MetricInfoArtifact = evaluate_regression_model(model_list=model_list, X_train=X_train,
                                                                        Y_train=Y_train, X_test=X_test, Y_test=Y_test,
                                                                        base_r2=base_r2)
            
            
            logging.info(f"-------------")
            
            logging.info(f"Model Selected : {metric_info.model_name}")
            logging.info(f"Best found model on both training and testing dataset.")
         
                        
            logging.info(f"-------------")

            preprocessing_obj = load_object(file_path=self.data_transformation_artifact.preprocessed_object_file_path)
            model_object = metric_info.model_object

            trained_model_file_path = self.model_trainer_config.trained_model_file_path
            predictor_result = Predictor(preprocessing_object=preprocessing_obj,
                                                      trained_model_object=model_object)
            
    
            logging.info(f"Saving model at path: {trained_model_file_path}")
            save_object(file_path=trained_model_file_path, obj=model_object)
          
    



            # Making Report 
            
            best_model_r2_score =str(metric_info.model_r2)
            # Model name 
            best_model_name = re.search(r"([^.]+)$", model_object.__class__.__name__).group(1)
            logging.info(f"Model Name : {best_model_name}")
            report={
                "Model_name":best_model_name,
                "R2_score":best_model_r2_score
            }
            
            logging.info(f"Dumping R2_Score in report.....")
             # Save report in artifact folder
            model_artifact_report_path = self.model_trainer_config.report_path
            with open(model_artifact_report_path, 'w') as file:
                yaml.safe_dump(report, file)
            logging.info("-----------------------")
            
            model_trainer_artifact = ModelTrainerArtifact(
                                                is_trained=True,
                                                message="Model Trained successfully",
                                                trained_model_file_path=trained_model_file_path,
                                                model_artifact_report=model_artifact_report_path,
                                                train_mse=metric_info.train_mse,
                                                test_mse=metric_info.test_mse,
                                                train_r2=metric_info.train_r2,
                                                test_r2=metric_info.test_r2,
                                                model_r2=metric_info.model_r2
                                            )
            

            logging.info(f"Model Trainer Artifact: {model_trainer_artifact}")
            return model_trainer_artifact
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def __del__(self):
        logging.info(f"{'>>' * 30}Model trainer log completed.{'<<' * 30} ")
            
            