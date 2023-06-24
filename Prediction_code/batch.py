import os
import logging
from Concrete_Strength_Prediction.logger import logging
from Concrete_Strength_Prediction.exception import ApplicationException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from Concrete_Strength_Prediction.utils.utils import read_yaml_file,load_object
from Concrete_Strength_Prediction.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
import sys 
import pymongo
import json
from Concrete_Strength_Prediction.constant import *
from Concrete_Strength_Prediction.constant import *
import urllib
import yaml
import numpy as np







class batch_prediction:
    def __init__(self,input_file_path, 
                 model_file_path, 
                 transformer_file_path, 
                 feature_engineering_file_path) -> None:
        
        self.input_file_path = input_file_path
        self.model_file_path = model_file_path
        self.transformer_file_path = transformer_file_path
        self.feature_engineering_file_path = feature_engineering_file_path

        
    
    def start_batch_prediction(self):
        try:
            logging.info("Loading the saved pipeline")

            # Load the feature engineering pipeline
            with open(self.feature_engineering_file_path, 'rb') as f:
                feature_pipeline = pickle.load(f)

            logging.info(f"Feature eng Object acessed :{self.feature_engineering_file_path}")
            
            
            # Load the data transformation pipeline
            with open(self.transformer_file_path, 'rb') as f:
                preprocessor = pickle.load(f)

            logging.info(f"Preprocessor  Object acessed :{self.transformer_file_path}")
            
            # Load the model separately
            model =load_object(file_path=self.model_file_path)

            logging.info(f"Model File Path: {self.model_file_path}")

            # Feature Labels
            transformation = read_yaml_file("config/transformation.yaml")
            input_features = transformation['numerical_column']
            target_features = transformation['target_column']
            
            # Drop columns 
            drop_columns = transformation['drop_columns']
            all_columns = input_features + target_features

            # Create the feature engineering pipeline
            feature_engineering_pipeline = Pipeline([
                ('feature_engineering', feature_pipeline)
            ])

            # Read the input file
            df = pd.read_csv(self.input_file_path)

            # Apply feature engineering
            array = feature_engineering_pipeline.transform(df)
            df = pd.DataFrame(array, columns=all_columns)

            # Save the feature-engineered data as a CSV file
            FEATURE_ENG_PATH = FEATURE_ENG  # Specify the desired path for saving the CSV file
            os.makedirs(FEATURE_ENG_PATH, exist_ok=True)
            file_path = os.path.join(FEATURE_ENG_PATH, 'batch_fea_eng.csv')
            df.to_csv(file_path, index=False)
            logging.info("Feature-engineered batch data saved as CSV.")
            
            # Dropping target column
            
            df=df.drop('Strength', axis=1)
            
            #df.to_csv('dropped_strength.csv')
          
                
                
            logging.info(f"Columns before transformation: {', '.join(f'{col}: {df[col].dtype}' for col in df.columns)}")
            # Transform the feature-engineered data using the preprocessor
            transformed_data = preprocessor.transform(df)
            logging.info(f"Transformed Data Shape: {transformed_data.shape}")
            
            logging.info(f"Loaded numpy from batch prediciton :{transformed_data}")
            file_path = os.path.join(FEATURE_ENG_PATH, 'preprocessor.csv')
            # Convert array to DataFrame with custom column names
            df = pd.DataFrame(transformed_data, columns=input_features)
            df.to_csv(file_path, index=False)
            
            
            logging.info(f"Model Data Type : {type(model)}")
            
            predictions = model.predict(transformed_data)
            logging.info(f"Predictions done :{predictions}")
            
            

            # Create a DataFrame from the predictions array
            df_predictions = pd.DataFrame(predictions, columns=['prediction'])
                        # Define the mapping dictionary
    
            # Save the predictions to a CSV file
            BATCH_PREDICTION_PATH = BATCH_PREDICTION  # Specify the desired path for saving the CSV file
            os.makedirs(BATCH_PREDICTION_PATH, exist_ok=True)
            csv_path = os.path.join(BATCH_PREDICTION_PATH,'predictions.csv')
            df_predictions.to_csv(csv_path, index=False)
            logging.info(f"Batch predictions saved to '{csv_path}'.")

        except Exception as e:
            ApplicationException(e,sys) 

