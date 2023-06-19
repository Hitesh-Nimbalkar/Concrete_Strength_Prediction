import os 
import sys
import pandas as pd
import numpy as np
from Concrete_Strength_Prediction.logger import logging
from Concrete_Strength_Prediction.exception import ApplicationException
from Concrete_Strength_Prediction.entity.artifact_entity import *
from Concrete_Strength_Prediction.entity.config_entity import *
from Concrete_Strength_Prediction.utils.utils import read_yaml_file,save_data,save_object
from Concrete_Strength_Prediction.constant import *

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import PowerTransformer

class Feature_Engineering(BaseEstimator, TransformerMixin):
    
    def __init__(self,numerical_columns,target_columns,drop_columns):
        
        """
        This class applies necessary Feature Engneering 
        """
        logging.info(f"\n{'*'*20} Feature Engneering Started {'*'*20}\n\n")
        

                                ############### Accesssing Column Labels #########################
                                
                                
                 #   Schema.yaml -----> Data Tranformation ----> Method: Feat Eng Pipeline ---> Class : Feature Eng Pipeline              #
                                
                                
        self.numerical_columns = numerical_columns
        self.target_columns = target_columns
        self.columns_to_drop = drop_columns

        
                                ########################################################################
        
        logging.info(f" Numerical Columns , Categorical Columns , Target Column initialised in Feature engineering Pipeline ")


    # Feature Engineering Pipeline 
    
    
    
    
                                ######################### Data Modification ############################
        
    def drop_rows_with_nan(self, X: pd.DataFrame):
        # Log the shape before dropping NaN values
        logging.info(f"Shape before dropping NaN values: {X.shape}")
        
        # Drop rows with NaN values
        X = X.dropna()
        #X.to_csv("Nan_values_removed.csv", index=False)
        
        # Log the shape after dropping NaN values
        logging.info(f"Shape after dropping NaN values: {X.shape}")
        
        logging.info("Dropped NaN values.")
        
        return X
 
    


    def data_type_modification(self,X): # Age int 
        
     
        # Categorical column from schema file
        object_cols = self.numerical_columns

        for col in object_cols:
            if col == 'AgeInDays':
                X[col] = X[col].astype('int64')
                
                

        return X
        
    def remove_duplicate_rows_keep_last(self,X):
        
        logging.info(f"DataFrame shape before removing duplicates: {X.shape}")
        num_before = len(X)
        X.drop_duplicates(inplace = True)
        num_after = len(X)
        
        num_duplicates = num_before - num_after
        logging.info(f"Removed {num_duplicates} duplicate rows")
        logging.info(f"DataFrame shape after removing duplicates: {X.shape}")
        
        return X


    def drop_columns(self,X:pd.DataFrame):
        try:
            columns=X.columns
            
            logging.info(f"Columns before drop  {columns}")
            
            # Columns Dropping
            drop_column_labels=self.columns_to_drop
            
            logging.info(f" Dropping Columns {drop_column_labels} ")
            
            X=X.drop(columns=drop_column_labels,axis=1)
            
            return X
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
    
    def run_data_modification(self,data):
    
        
        # Drop Columns 
        X=self.drop_columns(X=data)
        
        
        # Drop rows with nan
        X=self.drop_rows_with_nan(X)
        
        # Removing duplicated rows 
        X=self.remove_duplicate_rows_keep_last(X)
        
        # Modifying datatype Object ---> categorical
        X=self.data_type_modification(X)
        
        # Drop rows with nan
        X=self.drop_rows_with_nan(X)
        
        return X
    
    
    
    
    
    
    
                                            ######################### Outiers ############################
    
    
    
    
    
    def detect_outliers(self, data):
        outliers = {}
        
        numeric_cols=self.numerical_columns
        
        # Loop through numeric columns
        for col in numeric_cols:
            # Calculate the lower and upper quantiles
            q1 = data[col].quantile(0.25)
            q3 = data[col].quantile(0.75)
            
            # Calculate the interquartile range (IQR)
            iqr = q3 - q1
            
            # Define the lower and upper bounds for outliers
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            # Detect outliers
            col_outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col]
            
            # Save outliers for the column
            outliers[col] = col_outliers
            
            # Logging
            logging.info(f"Detected {len(col_outliers)} outliers in column '{col}'.")
        
        return outliers

    def outliers_replaced(self, data, outliers):
        # Loop through columns with outliers
        for col, col_outliers in outliers.items():
            # Calculate the median for the column
            col_median = data[col].median()
            
            # Replace outliers with the median
            data.loc[data[col].isin(col_outliers), col] = col_median
            
            # Logging
            logging.info(f"Replaced {len(col_outliers)} outliers in column '{col}' with median value: {col_median}.")
        
        return data
    
    def outlier(self,X):
        
        outliers=self.detect_outliers(X)
        X=self.outliers_replaced(X,outliers)

        return X
    
    
    
    
    
    def data_wrangling(self,X:pd.DataFrame):
        try:

            
            # Data Modification 
            data_modified=self.run_data_modification(data=X)
            
            logging.info(" Data Modification Done")
            
            # Outlier Detection and Removal
            outliers_removed:pd.DataFrame = self.outlier(data_modified)
            
           # outliers_removed.to_csv("outliers_removed.csv",index=False)
            
            logging.info(" Outliers detection and removal Done ")
            
            

            
            return outliers_removed
    
        
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
        
    
    
    
    
    def fit(self,X,y=None):
        return self
    
    
    def transform(self,X:pd.DataFrame,y=None):
        try:    
            data_modified=self.data_wrangling(X)
            
            numerical_columns = self.numerical_columns
            target_column=self.target_columns
            
            
            col = numerical_columns+target_column
            
            
            print("\n")
            logging.info(f"New Column Order {col}")
            print("\n")
            
            
            data_modified:pd.DataFrame = data_modified[col]
            
            data_modified.to_csv("data_modified.csv",index=False)
            logging.info(" Data Wrangaling Done ")
            
            logging.info(f"Original Data  : {X.shape}")
            logging.info(f"Shapde Modified Data : {data_modified.shape}")
         
            
            arr = data_modified.values
                
            return arr
        except Exception as e:
            raise ApplicationException(e,sys) from e







class DataTransformation:
    
    
    def __init__(self, data_transformation_config: DataTransformationConfig,
                    data_validation_artifact: DataValidationArtifact):
        try:
            logging.info(f"\n{'*'*20} Data Transformation log started {'*'*20}\n\n")
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
            
                                ############### Accesssing Column Labels #########################
                                
                                
                                #           Schema.yaml -----> DataTransfomation 
            
            # Transformation Yaml File path 
            
            # Reading data in Schema 
            self.transformation_yaml = read_yaml_file(file_path=TRANSFORMATION_YAML_FILE_PATH)
            
            # Column data accessed from Schema.yaml
            self.target_column_name = self.transformation_yaml[TARGET_COLUMN_KEY]
            self.numerical_columns = self.transformation_yaml[NUMERICAL_COLUMN_KEY] 
            self.drop_columns=self.transformation_yaml[DROP_COLUMNS]
            
           # self.drop_columns=self.schema[DROP_COLUMN_KEY]
            
                                ########################################################################
        except Exception as e:
            raise ApplicationException(e,sys) from e



    def get_feature_engineering_object(self):
        try:
            
            feature_engineering = Pipeline(steps = [("fe",Feature_Engineering(numerical_columns=self.numerical_columns,
                                                                            target_columns=self.target_column_name,
                                                                            drop_columns=self.drop_columns))])
            return feature_engineering
        except Exception as e:
            raise ApplicationException(e,sys) from e
        
    def get_data_transformer_object(self):
        try:
            logging.info('Creating Data Transformer Object')
            numerical_col=self.numerical_columns

            numerical_pipeline = Pipeline(steps=[
                ('impute', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler()),
                ('power', PowerTransformer(method='yeo-johnson', standardize=False))
            ])
            preprocessor = ColumnTransformer([
                ('numerical_pipeline', numerical_pipeline, numerical_col)
            ])
            return preprocessor


        except Exception as e:
                logging.error('An error occurred during data transformation')
                raise ApplicationException(e, sys) from e
            




    def initiate_data_transformation(self):
        try:
            # Data validation Artifact ------>Accessing train and test files 
            logging.info(f"Obtaining training and test file path.")
            train_file_path = self.data_validation_artifact.validated_train_path
            test_file_path = self.data_validation_artifact.validated_test_path

            logging.info(f"Loading training and test data as pandas dataframe.")
            train_df = pd.read_csv(train_file_path)
            test_df = pd.read_csv(test_file_path)
            
            logging.info(f" Traning columns {train_df.columns}")
            
            # Schema.yaml ---> Extracting target column name
            target_column_name = self.target_column_name
            numerical_columns = self.numerical_columns
            drop_columns=self.drop_columns
                        
            # Log column information
            logging.info("Numerical columns: {}".format(numerical_columns))
            logging.info("Target Column: {}".format(target_column_name))
            
            
            col = numerical_columns + target_column_name
            # All columns 
            logging.info("All columns: {}".format(col))
            
            
            # Feature Engineering 
            logging.info(f"Obtaining feature engineering object.")
            fe_obj = self.get_feature_engineering_object()
            
            logging.info(f"Applying feature engineering object on training dataframe and testing dataframe")
            logging.info(">>>" * 20 + " Training data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Train Data ")
            feature_eng_train_arr = fe_obj.fit_transform(train_df)
            logging.info(">>>" * 20 + " Test data " + "<<<" * 20)
            logging.info(f"Feature Enineering - Test Data ")
            feature_eng_test_arr = fe_obj.transform(test_df)
            
            
            
            # Converting featured engineered array into dataframe
            logging.info(f"Converting featured engineered array into dataframe.")
            
            feature_eng_train_df = pd.DataFrame(feature_eng_train_arr,columns=col)
            #feature_eng_train_df.to_csv('feature_eng_train_df.csv',index=False)
            
            logging.info(f"Feature Engineering - Train Completed")
            
            feature_eng_test_df = pd.DataFrame(feature_eng_test_arr,columns=col)
            #feature_eng_test_df.to_csv('feature_eng_test_df.csv',index=False)
            
            #logging.info(f" Columns in feature enginering test {feature_eng_test_df.columns}")
            logging.info(f"Saving feature engineered training and testing dataframe.")
            
            
            
            # Train and Test Dataframe
            target_column_name=self.target_column_name

            target_feature_train_df = feature_eng_train_df[target_column_name]
            input_feature_train_df = feature_eng_train_df.drop(columns = target_column_name,axis = 1)
             
            target_feature_test_df = feature_eng_test_df[target_column_name]
            input_feature_test_df = feature_eng_test_df.drop(columns = target_column_name,axis = 1)
            
                                            ######## TARGET COLUMN ##########

  
                                                #############################
                        
                                    ############ Input Fatures transformation########
            ## Preprocessing 
            logging.info("*" * 20 + " Applying preprocessing object on training dataframe and testing dataframe " + "*" * 20)
            preprocessing_obj = self.get_data_transformer_object()

            col = numerical_columns + target_column_name

            train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            test_arr = preprocessing_obj.transform(input_feature_test_df)
            
            
            # Log the shape of train_arr
            logging.info(f"Shape of train_arr: {train_arr.shape}")

            # Log the shape of test_arr
            logging.info(f"Shape of test_arr: {test_arr.shape}")

            logging.info("Transformation completed successfully")
            
            
            col =numerical_columns
            
            

            train_arr = np.c_[train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[test_arr, np.array(target_feature_test_df)]
            transformed_train_df = pd.DataFrame(train_arr, columns=col + [target_column_name])
            transformed_test_df = pd.DataFrame(test_arr, columns=col + [target_column_name])
            
            # Adding target column to transformed dataframe
            transformed_train_dir = self.data_transformation_config.transformed_train_dir
            transformed_test_dir = self.data_transformation_config.transformed_test_dir    
        
            transformed_train_file_path = os.path.join(transformed_train_dir,"transformed_train.csv")
            transformed_test_file_path = os.path.join(transformed_test_dir,"transformed_test.csv")

                    

                                ###############################################################
            
            # Saving the Transformed array ----> csv 
            ## Saving transformed train and test file
            logging.info("Saving Transformed Train and Transformed test file")
            
            save_data(file_path = transformed_train_file_path, data = transformed_train_df)
            save_data(file_path = transformed_test_file_path, data = transformed_test_df)
            logging.info("Transformed Train and Transformed test file saved")
            logging.info("Saving Feature Engineering Object")
            
            ### Saving FFeature engineering and preprocessor object 
            logging.info("Saving Feature Engineering Object")
            feature_engineering_object_file_path = self.data_transformation_config.feature_engineering_object_file_path
            save_object(file_path = feature_engineering_object_file_path,obj = fe_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(feature_engineering_object_file_path)),obj=fe_obj)

            logging.info("Saving Preprocessing Object")
            preprocessing_object_file_path = self.data_transformation_config.preprocessed_object_file_path
            save_object(file_path = preprocessing_object_file_path, obj = preprocessing_obj)
            save_object(file_path=os.path.join(ROOT_DIR,PIKLE_FOLDER_NAME_KEY,
                                 os.path.basename(preprocessing_object_file_path)),obj=preprocessing_obj)

            data_transformation_artifact = DataTransformationArtifact(is_transformed=True,
            message="Data transformation successfull.",
            transformed_train_file_path = transformed_train_file_path,
            transformed_test_file_path = transformed_test_file_path,
            preprocessed_object_file_path = preprocessing_object_file_path,
            feature_engineering_object_file_path = feature_engineering_object_file_path)
            
            logging.info(f"Data Transformation Artifact: {data_transformation_artifact}")
            return data_transformation_artifact
        except Exception as e:
            raise ApplicationException(e,sys) from e

    def __del__(self):
        logging.info(f"\n{'*'*20} Data Transformation log completed {'*'*20}\n\n")