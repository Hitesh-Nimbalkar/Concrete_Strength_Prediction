            
            
import yaml           
import shutil
import os           
import sys 
from Concrete_Strength_Prediction.logger import logging
from Concrete_Strength_Prediction.exception import ApplicationException           
from Concrete_Strength_Prediction.entity.artifact_entity import ModelEvaluationArtifact,ModelPusherArtifact
from Concrete_Strength_Prediction.utils.utils import load_object,save_object
from Concrete_Strength_Prediction.constant.training_pipeline import *     
from Concrete_Strength_Prediction.constant import *       

            
            
            
            
          
class ModelPusher:

    def __init__(self,model_eval_artifact:ModelEvaluationArtifact):

        try:
            self.model_eval_artifact = model_eval_artifact
        except  Exception as e:
            raise ApplicationException(e, sys)
      
            
            
            
    def initiate_model_pusher(self):
        try:
            # Selected model path
            model_path = self.model_eval_artifact.selected_model_path
            logging.info(f" Model path : {model_path}")
            model = load_object(file_path=model_path)
            file_path=os.path.join(ROOT_DIR,SAVED_MODEL_DIRECTORY,'model.pkl')
            
            save_object(file_path=file_path, obj=model)
            logging.info("Model saved.")
            
            # Model report
            model_name = self.model_eval_artifact.model_name
            R2_score = self.model_eval_artifact.R2_score
           
            
            
            # Create a dictionary for the report
            report = {'Model_name': model_name, 'R2_score': R2_score}

            logging.info(str(report))
            
            # Save the report as a YAML file
            file_path=os.path.join(ROOT_DIR,SAVED_MODEL_DIRECTORY,MODEL_REPORT_FILE)
            logging.info(f"Report Location: {file_path}")

            # Save the report as a YAML file
            with open(file_path, 'w') as file:
                yaml.dump(report, file)

            logging.info("Report saved as YAML file.")
            
            
        

            model_pusher_artifact = ModelPusherArtifact(message="Model Pushed succeessfully")
            return model_pusher_artifact
        except  Exception as e:
            raise ApplicationException(e, sys)
    
            
            
    def __del__(self):
        logging.info(f"\n{'*'*20} Model Pusher log completed {'*'*20}\n\n")
            
            
            
            
            
            
            
            
            
            
 