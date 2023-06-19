'''
    def get_and_save_data_drift_report(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            logging.info("Generating data drift report.json file")
            profile = Profile(sections=[DataDriftProfileSection()])
            profile.calculate(train_df, test_df)
            report_file_path = self.data_validation_config.drift_report
            print(f"{report_file_path}")
            report = json.loads(profile.json())

            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)
            with open(report_file_path, "w") as report_file:
                json.dump(report, report_file, indent=6)
            logging.info("Report.json file generation successful!!")
            return report
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
        
  

    def get_and_save_data_drift_report(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        try:
            logging.info("Generating data drift report.json file")
            
            # Perform data drift analysis using pandas_profiling
            report = pandas_profiling.compare(train_df, test_df)
            
            report_file_path = self.data_validation_config.drift_report
            print(f"{report_file_path}")

            report_dir = os.path.dirname(report_file_path)
            os.makedirs(report_dir, exist_ok=True)
            
            # Save the report as JSON
            report.to_file(report_file_path)
            
            logging.info("Report.json file generation successful!!")
            return report
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def save_data_drift_report_page(self,train_df,test_df):
        try:
            logging.info("Generating data drift report.html page")
            dashboard = Dashboard(tabs = [DataDriftTab()])
            dashboard.calculate(train_df, test_df)

            report_page_file_path = self.data_validation_config.drift_report_page
            report_page_dir = os.path.dirname(report_page_file_path)
            os.makedirs(report_page_dir,exist_ok=True)

            dashboard.save(report_page_file_path)
            logging.info("Report.html page generation successful!!")
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def is_data_drift_found(self) -> bool:
        try:
            logging.info("Checking for Data Drift")
            train_df=pd.read_csv(self.data_ingestion_artifact.train_file_path)
            test_df=pd.read_csv(self.data_ingestion_artifact.test_file_path)
            report = self.get_and_save_data_drift_report(train_df=train_df,test_df=test_df)
            self.save_data_drift_report_page(train_df=train_df,test_df=test_df)
            return True
        except Exception as e:
            raise ApplicationException(e,sys) from e


import pandas_profiling
from evidently.model_profile import Profile
from evidently.model_profile.sections import DataDriftProfileSection
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab




        '''


import pandas as pd
from Concrete_Strength_Prediction.exception import ApplicationException
from Concrete_Strength_Prediction.logger import logging
import os
import pandas_profiling
import sys
from Concrete_Strength_Prediction.entity.artifact_entity import *




logging.info("Checking for Data Drift")
train_df=pd.read_csv(r'E:\Ineuron may batch projects\Projects\Concrete strength\train.csv')
print(train_df.head())
test_df=pd.read_csv(r'E:\Ineuron may batch projects\Projects\Concrete strength\test.csv')


def get_and_save_data_drift_report( train_df: pd.DataFrame, test_df: pd.DataFrame):
    try:
        logging.info("Generating data drift report.json file")
        print(train_df.head())
        # Perform data drift analysis using pandas_profiling
        report = pandas_profiling.compare(train_df, test_df)
        
        print(report)
        
        report_file_path = os.getcwd()
        print(f"{report_file_path}")

        report_dir = os.path.dirname(report_file_path)
        os.makedirs(report_dir, exist_ok=True)
        
        # Save the report as JSON
        report.to_file(report_file_path)
        
        logging.info("Report.json file generation successful!!")
        return report
    except Exception as e:
        raise ApplicationException(e, sys) from e
    
report = get_and_save_data_drift_report(train_df=train_df,test_df=test_df)

##############################################################################################


import pandas as pd
import numpy as np

def calculate_data_drift(train_df, test_df):
    # Calculate the absolute difference in means for each numeric column
    numeric_columns = train_df.select_dtypes(include=np.number).columns
    means_train = train_df[numeric_columns].mean()
    means_test = test_df[numeric_columns].mean()
    mean_diff = abs(means_train - means_test)
    
    # Calculate the absolute difference in value counts for each categorical column
    categorical_columns = train_df.select_dtypes(include='object').columns
    value_counts_train = train_df[categorical_columns].apply(lambda x: x.value_counts().sort_index())
    value_counts_test = test_df[categorical_columns].apply(lambda x: x.value_counts().sort_index())
    value_counts_diff = abs(value_counts_train - value_counts_test)
    
    # Calculate the overall data drift score
    data_drift_score = mean_diff.mean() + value_counts_diff.mean().mean()
    
    return data_drift_score