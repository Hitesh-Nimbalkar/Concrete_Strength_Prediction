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
from Prediction_code.predict_dump import prediction_upload
from Concrete_Strength_Prediction.pipeline.train import Pipeline

feature_engineering_file_path = "Prediction_Files/feat_eng.pkl"
transformer_file_path = "Prediction_Files/preprocessed.pkl"
model_file_path = "Saved_model/model.pkl"

UPLOAD_FOLDER = 'batch_prediction/Uploaded_CSV_FILE'

app = Flask(__name__)
ALLOWED_EXTENSIONS = {'csv'}

@app.route("/", methods=["GET", "POST"])
def home():
    return render_template("index.html")

@app.route("/batch", methods=["GET","POST"])
def perform_batch_prediction():
    if request.method == 'GET':
        return render_template('batch.html')
    else:
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
            batch = batch_prediction(file_path, model_file_path, transformer_file_path, feature_engineering_file_path)
            batch.start_batch_prediction()

            # ----------------------------------------------- Uploading the Prediction.csv ---------------------------------------#
            
            # Get the file from the request
            prediction_file = 'batch_prediction/prediction_csv/predictions.csv'

            Database_name_default = DATABASE_NAME
            Collection_name_default_prediction = 'prediction'

            # Get the user-provided database and collection names, or use default values
            DATABASE_NAME_PREDICTION = request.form.get('database_name_prediction', Database_name_default)
            COLLECTION_NAME_PREDICTION = request.form.get('collection_name_prediction', Collection_name_default_prediction)

            # Create an instance of data_dump_prediction
            data_dumper = prediction_upload(DATABASE_NAME_PREDICTION, COLLECTION_NAME_PREDICTION)

            # Call the data_dump method with the uploaded file
            data_dumper.data_dump(filepath=prediction_file)
           

            output = "Batch Prediction Done and Uploaded on Mongo Database"
            return render_template("batch.html", prediction_result=output,prediction_type='batch')
        else:
            return render_template('batch.html', prediction_type='batch', error='Invalid file type')
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('instance.html')
    else:
        # Extract the input features from the form data
        age = float(request.form.get('age'))
        slag = float(request.form.get('slag'))
        cement = float(request.form.get('cement'))
        superplasticizer = float(request.form.get('superplasticizer'))
        water = float(request.form.get('water'))

        logging.info("All data taken")

        # Create an instance of the instance_prediction_class
        predictor = instance_prediction_class(age, slag, cement, superplasticizer, water)

        predicted_strength = predictor.predict_price_from_input()
        logging.info("Prediction done")

        return render_template('instance.html', prediction_type='instance', predicted_strength=predicted_strength)



@app.route('/train', methods=['GET'])
def train():
    try:
        pipeline = Pipeline()
        pipeline.run_pipeline()

        return render_template('index.html', message="Training complete")

    except Exception as e:
        logging.error(f"{e}")
        error_message = str(e)
        return render_template('index.html', error=error_message)

if __name__ == '__main__':
    host = '0.0.0.0'  # Specify the host address you want to use
    port = 8080  # Specify the port number you want to use
    app.run(debug=True, host=host, port=port)
