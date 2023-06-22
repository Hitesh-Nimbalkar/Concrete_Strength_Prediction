from Prediction_code.instance import instance_prediction_class
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from Prediction_code.batch import batch_prediction
from Prediction_code.instance import instance_prediction_class
import os
from Concrete_Strength_Prediction.logger import logging
from Concrete_Strength_Prediction.constant import *
from Concrete_Strength_Prediction.entity.artifact_entity import ModelEvaluationArtifact
from Concrete_Strength_Prediction.constant import *


feature_engineering_file_path = "Prediction_Files/feat_eng.pkl"
transformer_file_path = "Prediction_Files/preprocessed.pkl"
model_file_path = "Saved_model/model.pkl"

UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'csv'}



@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/batch", methods=["POST"])
def perform_batch_prediction():
    file = request.files['csv_file']  # Update the key to 'csv_file'
        # Directory path
    directory_path = UPLOAD_FOLDER
    # Create the directory
    os.makedirs(directory_path, exist_ok=True)

    
    # Check if the file has a valid extension
    if file and '.' in file.filename and file.filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS:
        # Delete all files in the file path
        for filename in os.listdir(os.path.join(UPLOAD_FOLDER)):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)

        # Save the new file to the uploads directory
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(file_path)
        print(file_path)
        
        logging.info("CSV received and Uploaded")

        # Perform batch prediction using the uploaded file
        batch = batch_prediction(file_path, 
                                 model_file_path, 
                                 transformer_file_path, 
                                 feature_engineering_file_path)
        batch.start_batch_prediction()

        output = "Batch Prediction Done"
        return render_template("index.html", prediction_result=output, prediction_type='batch')
    else:
        return render_template('index.html', prediction_type='batch', error='Invalid file type')



@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Get the JSON data from the request

    # Extract the input features from the JSON data
    age = data['AgeInDays']
    slag = data['BlastFurnaceSlag']
    cement = data['CementComponent']
    superplasticizer = data['SuperplasticizerComponent']
    water = data['WaterComponent']

    # Create an instance of the instance_prediction_class
    predictor = instance_prediction_class(age, slag, cement, superplasticizer, water)

    predicted_strength = predictor.predict_price_from_input()


    return render_template('index.html', prediction_type='instance', predicted_strength=predicted_strength)

if __name__ == '__main__':
    host = '0.0.0.0'  # Specify the host address you want to use
    port = 8080  # Specify the port number you want to use
    app.run(debug=True, host=host, port=port)