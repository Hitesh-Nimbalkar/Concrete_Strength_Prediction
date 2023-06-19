import os
import sys
import numpy as np
import pandas as pd
def create_folder(path):
    try:
        os.makedirs(path)
        print(f"Created folder: {path}")
    except FileExistsError:
        print(f"Folder already exists: {path}")

def create_file(path): 
    try:
        with open(path, 'w') as file:
            print(f"Created file: {path}")
    except FileExistsError:
        print(f"File already exists: {path}")

# Define your desired folder and file structure here
def create_template():
    # Prompt user for project title
        # Create files
    create_file('main.py')
    create_file('app.py')
    create_file('setup.py')
    create_file('README.md')
    create_file('env.yaml')
    create_file('requirements.txt')
    create_file('data_dump.py')
    create_file('.gitignore')
    project_title = input("Enter the project title: ")

    # Create subfolders
    create_folder(project_title)
    create_folder(f"{project_title}/configuration")
    create_folder(f"{project_title}/constant")
    create_folder(f"{project_title}/constant/training_pipeline")
    create_folder(f"{project_title}/data_access")
    create_folder(f"{project_title}/entity")
    create_folder(f"{project_title}/pipeline")
    create_folder(f"{project_title}/utils")
    create_folder(f"{project_title}/components")

    # Create __init__.py files in subfolders
    create_file(f"{project_title}/__init__.py")
    create_file(f"{project_title}/configuration/__init__.py")
    create_file(f"{project_title}/constant/__init__.py")
    create_file(f"{project_title}/constant/training_pipeline/__init__.py")
    create_file(f"{project_title}/data_access/__init__.py")
    create_file(f"{project_title}/entity/__init__.py")
    create_file(f"{project_title}/pipeline/__init__.py")
    create_file(f"{project_title}/utils/__init__.py")

    # Create Python files in the components folder
    create_file(f"{project_title}/components/data_ingestion.py")
    def def_data_ingestion():
        code = '''
        from {project_title}.exception import ApplicationException
        from {project_title}.logger import logging
        from {project_title}.entity.config_entity import DataIngestionConfig
        from {project_title}.entity.artifact_entity import DataIngestionArtifact
        from sklearn.model_selection import train_test_split
        import os, sys
        from pandas import DataFrame
        from {project_title}.data_access.sensor_data import SensorData
        from {project_title}.utils.main_utils import read_yaml_file
        from {project_title}.constant.training_pipeline import SCHEMA_FILE_PATH
        import sys


        class DataIngestion:
            def __init__(self, data_ingestion_config: DataIngestionConfig):
                try:
                    self.data_ingestion_config = data_ingestion_config
                    self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
                except Exception as e:
                    raise ApplicationException(e, sys)

            def export_data_into_feature_store(self) -> DataFrame:
                """
                Export mongo db collection record as data frame into feature
                """
                try:
                    logging.info("Exporting data from mongodb to feature store")
                    sensor_data = SensorData()
                    dataframe = sensor_data.export_collection_as_dataframe(
                        collection_name=self.data_ingestion_config.collection_name
                    )
                    feature_store_file_path = self.data_ingestion_config.feature_store_file_path

                    # creating folder
                    dir_path = os.path.dirname(feature_store_file_path)
                    os.makedirs(dir_path, exist_ok=True)
                    dataframe.to_csv(feature_store_file_path, index=False, header=True)

                    logging.info("Exported dataframe")
                    return dataframe

                except Exception as e:
                    raise ApplicationException(e, sys)

            def split_data_as_train_test(self, dataframe: DataFrame) -> None:
                """
                Feature store dataset will be split into train and test file
                """

                try:
                    train_set, test_set = train_test_split(
                        dataframe, test_size=self.data_ingestion_config.train_test_split_ratio
                    )

                    logging.info("Performed train test split on the dataframe")

                    logging.info(
                        "Exited split_data_as_train_test method of Data_Ingestion class"
                    )

                    dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)

                    os.makedirs(dir_path, exist_ok=True)

                    logging.info(f"Exporting train and test file path.")

                    train_set.to_csv(
                        self.data_ingestion_config.training_file_path, index=False, header=True
                    )

                    test_set.to_csv(
                        self.data_ingestion_config.testing_file_path, index=False, header=True
                    )

                    logging.info(f"Exported train and test file path.")
                except Exception as e:
                    raise SensorData(e, sys)

            def initiate_data_ingestion(self) -> DataIngestionArtifact:
                try:
                    dataframe = self.export_data_into_feature_store()
                    dataframe = dataframe.drop(
                        self._schema_config["drop_columns"], axis=1
                    )
                    self.split_data_as_train_test(dataframe=dataframe)
                    data_ingestion_artifact = DataIngestionArtifact(
                        trained_file_path=self.data_ingestion_config.training_file_path,
                        test_file_path=self.data_ingestion_config.testing_file_path,
                    )
                    return data_ingestion_artifact
                except Exception as e:
                    raise ApplicationException(e, sys)
        '''

        return code
    code = def_data_ingestion()
    # Write the code to data_ingestion.py file
    with open(f'{project_title}/components/data_ingestion.py', 'w') as file:
        file.write(code)
    print("data_ingestion.py file created successfully.")

    #dsnlkdlkdss

    ## Code to data validation 
    create_file(f"{project_title}/components/data_validation.py")
    def data_validation():
        code = '''from distutils import dir_util
    from sensor.constant.training_pipeline import SCHEMA_FILE_PATH
    from sensor.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
    from sensor.entity.config_entity import DataValidationConfig
    from sensor.exception import ApplicationException
    from sensor.logger import logging
    from sensor.utils.main_utils import read_yaml_file,write_yaml_file
    from scipy.stats import ks_2samp
    import pandas as pd
    import os,sys
    class DataValidation:

        def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                            data_validation_config:DataValidationConfig):
            try:
                self.data_ingestion_artifact=data_ingestion_artifact
                self.data_validation_config=data_validation_config
                self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)
            except Exception as e:
                raise  ApplicationException(e,sys)
        
        def drop_zero_std_columns(self,dataframe):
            pass


        def validate_number_of_columns(self,dataframe:pd.DataFrame)->bool:
            try:
                number_of_columns = len(self._schema_config["columns"])
                logging.info(f"Required number of columns: {number_of_columns}")
                logging.info(f"Data frame has columns: {len(dataframe.columns)}")
                if len(dataframe.columns)==number_of_columns:
                    return True
                return False
            except Exception as e:
                raise ApplicationException(e,sys)

        def is_numerical_column_exist(self,dataframe:pd.DataFrame)->bool:
            try:
                numerical_columns = self._schema_config["numerical_columns"]
                dataframe_columns = dataframe.columns

                numerical_column_present = True
                missing_numerical_columns = []
                for num_column in numerical_columns:
                    if num_column not in dataframe_columns:
                        numerical_column_present=False
                        missing_numerical_columns.append(num_column)
                
                logging.info(f"Missing numerical columns: [{missing_numerical_columns}]")
                return numerical_column_present
            except Exception as e:
                raise ApplicationException(e,sys)

        @staticmethod
        def read_data(file_path)->pd.DataFrame:
            try:
                return pd.read_csv(file_path)
            except Exception as e:
                raise ApplicationException(e,sys)
        

        def detect_dataset_drift(self,base_df,current_df,threshold=0.05)->bool:
            try:
                status=True
                report ={}
                for column in base_df.columns:
                    d1 = base_df[column]
                    d2  = current_df[column]
                    is_same_dist = ks_2samp(d1,d2)
                    if threshold<=is_same_dist.pvalue:
                        is_found=False
                    else:
                        is_found = True 
                        status=False
                    report.update({column:{
                        "p_value":float(is_same_dist.pvalue),
                        "drift_status":is_found
                        
                        }})
                
                drift_report_file_path = self.data_validation_config.drift_report_file_path
                
                #Create directory
                dir_path = os.path.dirname(drift_report_file_path)
                os.makedirs(dir_path,exist_ok=True)
                write_yaml_file(file_path=drift_report_file_path,content=report,)
                return status
            except Exception as e:
                raise ApplicationException(e,sys)
    

        def initiate_data_validation(self)->DataValidationArtifact:
            try:
                error_message = ""
                train_file_path = self.data_ingestion_artifact.trained_file_path
                test_file_path = self.data_ingestion_artifact.test_file_path

                #Reading data from train and test file location
                train_dataframe = DataValidation.read_data(train_file_path)
                test_dataframe = DataValidation.read_data(test_file_path)

                #Validate number of columns
                status = self.validate_number_of_columns(dataframe=train_dataframe)
                if not status:
                    error_message=f"{error_message}Train dataframe does not contain all columns.\n"
                status = self.validate_number_of_columns(dataframe=test_dataframe)
                if not status:
                    error_message=f"{error_message}Test dataframe does not contain all columns.\n"
            

                #Validate numerical columns

                status = self.is_numerical_column_exist(dataframe=train_dataframe)
                if not status:
                    error_message=f"{error_message}Train dataframe does not contain all numerical columns.\n"
                
                status = self.is_numerical_column_exist(dataframe=test_dataframe)
                if not status:
                    error_message=f"{error_message}Test dataframe does not contain all numerical columns.\n"
                
                if len(error_message)>0:
                    raise Exception(error_message)

                #Let check data drift
                status = self.detect_dataset_drift(base_df=train_dataframe,current_df=test_dataframe)

                data_validation_artifact = DataValidationArtifact(
                    validation_status=status,
                    valid_train_file_path=self.data_ingestion_artifact.trained_file_path,
                    valid_test_file_path=self.data_ingestion_artifact.test_file_path,
                    invalid_train_file_path=None,
                    invalid_test_file_path=None,
                    drift_report_file_path=self.data_validation_config.drift_report_file_path,
                )

                logging.info(f"Data validation artifact: {data_validation_artifact}")

                return data_validation_artifact
            except Exception as e:
                raise ApplicationException(e,sys)'''
        return code 

    code = data_validation()
    # Write the code to data_ingestion.py file
    with open(f"{project_title}/components/data_validation.py", 'w') as file:
        file.write(code)
    print("data_validation.py file created successfully.")
    

    create_file(f"{project_title}/components/data_transformation.py")
    create_file(f"{project_title}/components/model_trainer.py")
    create_file(f"{project_title}/components/model_evaluation.py")
    create_file(f"{project_title}/components/model_pusher.py")

    # Create Python files in the Project_Title folder
    create_file(f"{project_title}/logger.py")
    def logger():
        code = '''import logging
from datetime import datetime
import os

LOG_DIR = "application_logs"

CURRENT_TIME_STAMP = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"

LOG_FILE_NAME = f"log_{CURRENT_TIME_STAMP}.log"

os.makedirs(LOG_DIR,exist_ok=True)

LOG_FILE_PATH = os.path.join(LOG_DIR,LOG_FILE_NAME)

logging.basicConfig(filename=LOG_FILE_PATH,
filemode="w",
level = logging.INFO,
format = '[%(asctime)s] %(name)s - %(levelname)s - %(message)s'
) '''
        return code
    code = logger()
    # Write the code to data_ingestion.py file
    with open(f"{project_title}/logger.py",'w') as file:
        file.write(code)
    print("logger.py file created successfully.")


    
    create_file(f"{project_title}/exception.py")
    def exception():
        code ='''from distutils.log import error
import os
import sys

class ApplicationException(Exception):
    
    def __init__(self, error_message:Exception, error_details:sys):
        super().__init__(error_message)
        self.error_message = ApplicationException.get_detailed_error_message(error_message=error_message,
                                                                                error_details=error_details)

    @staticmethod
    def get_detailed_error_message(error_message:Exception, error_details:sys)->str:
        """
        error_message: Exception object
        error_details: object of sys module
        """

        _, _, exec_tb = error_details.exc_info()

        line_number = exec_tb.tb_lineno
        file_name = exec_tb.tb_frame.f_code.co_filename

        error_message = f"""
        Error occured in script: [{file_name}] at 
        line number: [{line_number}] 
        error message: [{error_message}]"""
        return error_message

    def __str__(self):
        return self.error_message

    def __repr__(self):
        return ApplicationException.__name__.str()'''
        
        return code
    code = exception()
    # Write the code to data_ingestion.py file
    with open(f"{project_title}/exception.py",'w') as file:
        file.write(code)
    print("logger.py file created successfully.")
    create_file(f"{project_title}/__init__.py")

    # Create Python files in the configuration folder
    create_file(f"{project_title}/configuration/mongo_db_connection.py")

    # Create Python files in the constant folder
    create_file(f"{project_title}/constant/database.py")
    create_file(f"{project_title}/constant/application.py")
    create_file(f"{project_title}/constant/env_variable.py")
    create_file(f"{project_title}/constant/s3_bucket.py")

    # Create Python files in the utils folder
    create_file(f"{project_title}/utils/utils.py")

    # Create Python files in the pipeline folder
    create_file(f"{project_title}/pipeline/train.py")

    # Create Python files in the data_access folder
    create_file(f"{project_title}/data_access/mongo_data.py")
    def mongo_data():
        code = '''import sys
from typing import Optional

import numpy as np
import pandas as pd
import json
from sensor.configuration.mongo_db_connection import MongoDBClient
from sensor.constant.database import DATABASE_NAME
from sensor.exception import ApplicationException





class mongodata:
    """
    This class help to export entire mongo db record as pandas dataframe
    """

    def __init__(self):
        try:
            
            
            self.mongo_client = MongoDBClient(database_name=DATABASE_NAME)
            print("Mongo Client initialized successfully.")

        except Exception as e:
            raise ApplicationException(e,sys)


    def save_csv_file(self,file_path ,collection_name: str, database_name: Optional[str] = None):
        try:
            data_frame=pd.read_csv(file_path)
            data_frame.reset_index(drop=True, inplace=True)
            records = list(json.loads(data_frame.T.to_json()).values())
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            collection.insert_many(records)
            return len(records)
        except Exception as e:
            raise ApplicationException(e, sys)


    def export_collection_as_dataframe(
        self, collection_name: str, database_name: Optional[str] = None) -> pd.DataFrame:
        try:
            """
            export entire collectin as dataframe:
            return pd.DataFrame of collection
            """
            if database_name is None:
                collection = self.mongo_client.database[collection_name]
            else:
                collection = self.mongo_client[database_name][collection_name]
            df = pd.DataFrame(list(collection.find()))

            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"], axis=1)

            df.replace({"na": np.nan}, inplace=True)

            return df

        except Exception as e:
            raise ApplicationException(e, sys)

        '''
    
    
    

# Call the function to create the template
create_template()



