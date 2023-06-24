
import os
import logging
from Concrete_Strength_Prediction.logger import logging
from Concrete_Strength_Prediction.exception import ApplicationException
import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from Concrete_Strength_Prediction.utils.utils import read_yaml_file
from Concrete_Strength_Prediction.entity.artifact_entity import ModelTrainerArtifact,DataTransformationArtifact
import sys 
from Concrete_Strength_Prediction.utils.utils import read_yaml_file,load_object


INSTANCE_PREDICTION="Instance_prediction"


feature_engineering_file_path ="Prediction_Files/feat_eng.pkl"
transformer_file_path ="Prediction_Files/preprocessed.pkl"
model_path ="Saved_model/model.pkl"






import pandas as pd
import joblib

# Load the preprocessor and machine learning model
preprocessor = load_object('Prediction_Files/preprocessed.pkl')
model = load_object('Saved_model/model.pkl')


class instance_prediction_class:
    def __init__(self, AgeInDays, 
                 BlastFurnaceSlag, 
                 CementComponent,
                 SuperplasticizerComponent, 
                 WaterComponent) -> None:
        self.AgeInDays = AgeInDays
        self.BlastFurnaceSlag = BlastFurnaceSlag
        self.CementComponent = CementComponent
        self.SuperplasticizerComponent = SuperplasticizerComponent
        self.WaterComponent = WaterComponent
 
    def preprocess_input(self):
        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'AgeInDays': [self.AgeInDays],
            'BlastFurnaceSlag': [self.BlastFurnaceSlag],
            'CementComponent': [self.CementComponent],
            'SuperplasticizerComponent': [self.SuperplasticizerComponent],
            'WaterComponent': [self.WaterComponent]
        })

        # Preprocess the user input using the preprocessor
        preprocessed_array = preprocessor.transform(user_input)

       
        

        # Return the preprocessed input as a DataFrame
        return preprocessed_array

    def predict_strength(self, preprocessed_input):
        # Make a prediction using the pre-trained model
        predicted_strength = model.predict(preprocessed_input)

        # Return the array of predicted prices
        return predicted_strength

    def predict_price_from_input(self):
        # Preprocess the input using the preprocessor
        preprocessed_array = self.preprocess_input()

        # Make a prediction using the pre-trained model
        predicted_strength = self.predict_strength(preprocessed_array)

        # Round off the predicted shipment prices to two decimal places
        predicted_strength = [round(strength, 2) for strength in predicted_strength]

        # Print the rounded predicted shipment prices
        
        print(f"The predicted Concrete Strength  is:  {predicted_strength[0]}")

        return predicted_strength[0]