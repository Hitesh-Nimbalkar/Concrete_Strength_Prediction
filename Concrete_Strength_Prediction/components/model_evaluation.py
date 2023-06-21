
from Concrete_Strength_Prediction.exception import ApplicationException
from Concrete_Strength_Prediction.logger import logging
import sys
import os
from Concrete_Strength_Prediction.entity.config_entity import *
from Concrete_Strength_Prediction.entity.artifact_entity import *
from Concrete_Strength_Prediction.constant import *
from Concrete_Strength_Prediction.configuration import Configuration
from Concrete_Strength_Prediction.utils.utils import read_yaml_file,load_object
from Concrete_Strength_Prediction.constant.training_pipeline import *

class ModelEvaluation:


    def __init__(self,
                    data_validation_artifact:DataValidationArtifact,
                    model_trainer_artifact:ModelTrainerArtifact):
        
        try:

            self.data_validation_artifact=data_validation_artifact
            self.model_trainer_artifact=model_trainer_artifact
            
            # Saved Model config 
            self.config=Configuration()
            self.saved_model_config=self.config.saved_model_config()
            
        except Exception as e:
            raise ApplicationException(e,sys)
        
        
        
    def initiate_model_evaluation(self) -> ModelEvaluationArtifact:
        try:
            logging.info(" Model Evaluation Started ")
            ## Artifact trained Model  files
            model_trained_artifact_path = self.model_trainer_artifact.trained_model_file_path
            model_trained_report = self.model_trainer_artifact.model_artifact_report
            
            # Saved Model files

            saved_model_path = self.saved_model_config.saved_model_file_path
            saved_model_report_path=self.saved_model_config.saved_report_file_path

                        
            logging.info(f" Artifact Trained model :")

            # Load the model evaluation report from the saved YAML file

            
            



            # Loading the models
            logging.info("Saved_models directory .....")
            os.makedirs(SAVED_MODEL_DIRECTORY,exist_ok=True)
            
            # Check if SAVED_MODEL_DIRECTORY is empty
            if not os.listdir(SAVED_MODEL_DIRECTORY):
                model_trained_report_data = read_yaml_file(file_path=model_trained_report)
                
                artifact_model_r2_score =float( model_trained_report_data['R2_score'])
                model_name = model_trained_report_data['Model_name']
                R2_score = artifact_model_r2_score
                # Artifact ----> Model, Model Report 
                model_path = model_trained_artifact_path
                model_report_path = model_trained_report
                
            else:
                saved_model_report_data = read_yaml_file(file_path=saved_model_report_path)
                model_trained_report_data = read_yaml_file(file_path=model_trained_report)
                
                
                saved_model = load_object(file_path=saved_model_path)
                artifact_model = load_object(file_path=model_trained_artifact_path)

                # Compare the F1_Scores and accuracy of the two models
                saved_model_r2_score = float(saved_model_report_data['R2_score'])

                artifact_model_R2_score =float(model_trained_report_data['R2_score'])

                # Compare the models and log the result
                if artifact_model_R2_score > saved_model_r2_score:
                    logging.info("Trained model outperforms the saved model!")
                    model_path = model_trained_artifact_path
                    model_report_path = model_trained_report
                    model_name = model_trained_report_data['Model_name']
                    R2_score = float( model_trained_report_data['R2_score'])
                    
                    logging.info(f"Model Selected : {model_name}")
                    logging.info(f"F1_Score : {R2_score}")
                  
                elif artifact_model_R2_score < saved_model_r2_score:
                    logging.info("Saved model outperforms the trained model!")
                    model_path = saved_model_path
                    model_report_path = saved_model_report_path
                    model_name = saved_model_report_data['Model_name']
                    R2_score = float( saved_model_report_data['R2_score'])
                    logging.info(f"Model Seelcted : {model_name}")

                    logging.info(f"R2_score : {R2_score}")
            
                else:
                    logging.info("Both models have the same F1_Score.")
                    model_path = saved_model_path
                    model_report_path = saved_model_report_path
                    model_name = saved_model_report_data['Model_name']
                    R2_score = float( saved_model_report_data['R2_score'])
                    logging.info(f"Model Selected : {model_name}")

                    logging.info(f"R2_score : {R2_score}")
                
                

            # Create a model evaluation artifact
            model_evaluation = ModelEvaluationArtifact(model_name=model_name, R2_score=R2_score,
                                                    selected_model_path=model_path, 
                                                    model_report_path=model_report_path)

            logging.info("Model evaluation completed successfully!")

            return model_evaluation
        except Exception as e:
            logging.error("Error occurred during model evaluation!")
            raise ApplicationException(e, sys) from e


    def __del__(self):
        logging.info(f"\n{'*'*20} Model evaluation log completed {'*'*20}\n\n")