import importlib
from pyexpat import model
import numpy as np
import yaml
from Concrete_Strength_Prediction.exception import ApplicationException
import os
import sys

from collections import namedtuple
from typing import List
from Concrete_Strength_Prediction.logger import logging
from sklearn.metrics import mean_squared_error,r2_score

GRID_SEARCH_KEY = 'grid_search'
MODULE_KEY = 'module'
CLASS_KEY = 'class'
PARAM_KEY = 'params'
MODEL_SELECTION_KEY = 'model_selection'
SEARCH_PARAM_GRID_KEY = "search_param_grid"

# model_serial_number we need to discused

InitializedModelDetail = namedtuple("InitializedModelDetail",
                                    ["model_serial_number", "model", "param_grid_search", "model_name"])

GridSearchedBestModel = namedtuple("GridSearchedBestModel", ["model_serial_number",
                                                             "model",
                                                             "best_model",
                                                             "best_parameters",
                                                             "best_score",
                                                             ])

BestModel = namedtuple("BestModel", ["model_serial_number",
                                     "model",
                                     "best_model",
                                     "best_parameters",
                                     "best_score", ])

MetricInfoArtifact = namedtuple("MetricInfoArtifact",["model_name",
                                                    "model_object",
                                                    "train_r2",
                                                    "test_r2",
                                                    "train_mse",
                                                    "test_mse",
                                                    "model_r2",
                                                    "index_number"])

from sklearn.metrics import r2_score

def evaluate_regression_model(model_list: list, X_train, Y_train, X_test,
                              Y_test ,base_r2: float = 0.6) -> MetricInfoArtifact:
    """
    Description:
    This function compares multiple regression models and returns the best model.
    Params:
    model_list: List of models
    X_train: Training dataset input feature
    y_train: Training dataset target feature
    X_test: Testing dataset input feature
    y_test: Testing dataset target feature
    return:
    It returns a named tuple MetricInfoArtifact.
    
    MetricInfoArtifact = namedtuple("MetricInfo",
                                    ["model_name", "model_object", "train_r2", "test_r2", "train_mse",
                                     "test_mse", "model_r2", "index_number"])
    """
    try:
        # Convertinng Dataframes to array 
        X_train = X_train.values
        y_train = Y_train.values
        X_test = X_test.values
        y_test = Y_test.values
        
        
        index_number = 0
        metric_info_artifact = None
        for model in model_list:
            model_name = str(model)
            start_index = model_name.find("<")
            end_index = model_name.find(">")
            if start_index != -1 and end_index != -1:
                model_name = model_name[start_index+1:end_index].strip()
            else:
                model_name = "Unknown"

            # Getting predictions for training and testing datasets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculating R2 score on training and testing datasets
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Calculating mean squared error on training and testing datasets
            train_mse = mean_squared_error(y_train, y_train_pred)
            test_mse = mean_squared_error(y_test, y_test_pred)

            # Calculating harmonic mean of train_r2 and test_r2
            model_r2 = (2 * (train_r2 * test_r2)) / (train_r2 + test_r2)
            diff_test_train_r2 = abs(test_r2 - train_r2)
            
            
            logging.info(f"Started evaluating model: [{type(model).__name__}]")

            # Logging all important metrics
            logging.info(f"Scores ")
            logging.info(f"Train R2 Score : {train_r2} \tTest R2 Score\tAverage R2 Score")
            logging.info(f"{train_r2}\t\t{test_r2}\t\t{model_r2}")

            logging.info(f" Mean Squared Error ")

            logging.info(f"Diff test train R2: [{diff_test_train_r2}], For Model: {model}")
            logging.info(f"Train MSE: [{train_mse}]")
            logging.info(f"Test MSE: [{test_mse}]")


            # If model_r2 is greater than base_r2 and the difference between test_r2 and train_r2 is within a certain threshold,
            # we accept the model as the best model
            if model_r2 >= base_r2 and diff_test_train_r2 < 0.10:
                base_r2 = model_r2
                metric_info_artifact = MetricInfoArtifact(model_name=model_name,
                                                          model_object=model,
                                                          train_r2=train_r2,
                                                          test_r2=test_r2,
                                                          train_mse=train_mse,
                                                          test_mse=test_mse,
                                                          model_r2=model_r2,
                                                          index_number=index_number)

                logging.info(f"Acceptable model found: {metric_info_artifact}.")
            index_number += 1

        if metric_info_artifact is None:
            logging.info("No model found with higher R2 score than the base R2 score.")
        return metric_info_artifact
    except Exception as e:
        raise ApplicationException(e, sys) from e



class ModelFactory:
    def __init__(self, model_config_path: str = None, ):
        try:
            self.config: dict = ModelFactory.read_params(model_config_path)

            self.grid_search_cv_module: str = self.config[GRID_SEARCH_KEY][MODULE_KEY]
            self.grid_search_class_name: str = self.config[GRID_SEARCH_KEY][CLASS_KEY]
            self.grid_search_property_data: dict = dict(self.config[GRID_SEARCH_KEY][PARAM_KEY])

            self.models_initialization_config: dict = dict(self.config[MODEL_SELECTION_KEY])

            self.initialized_model_list = None
            self.grid_searched_best_model_list = None 

        except Exception as e:
            raise ApplicationException(e, sys) from e

    @staticmethod
    def update_property_of_class(instance_ref: object, property_data: dict):
        try:
            if not isinstance(property_data, dict):
                raise Exception("property_data parameter required to dictionary")
            print(property_data)
            for key, value in property_data.items():
               # logging.info(f"Executing:$ {str(instance_ref)}.{key}={value}")
                setattr(instance_ref, key, value)
            return instance_ref
        except Exception as e:
            raise ApplicationException(e, sys) from e
        
    # Read complete parameter from modelyaml file
    @staticmethod
    def read_params(config_path: str) -> dict:
        try:
            with open(config_path) as yaml_file:
                config: dict = yaml.safe_load(yaml_file)
            return config
        except Exception as e:
            raise ApplicationException(e, sys) from e



    # we defidning the class to call our model file randome forest and  KNN
    @staticmethod
    def class_for_name(module_name: str, class_name: str):
        try:
            # load the module, will raise ImportError if module cannot be loaded
            module = importlib.import_module(module_name)
            # get the class, will raise AttributeError if class cannot be found
            logging.info(f">>>>>>>>>>>>Executing command: from {module} import {class_name}<<<<<<<<<<<<")
            class_ref = getattr(module, class_name)
            return class_ref
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def execute_grid_search_operation(self, initialized_model: InitializedModelDetail, input_feature,
                                      output_feature) -> GridSearchedBestModel:
        """
        execute_grid_search_operation(): function will perform parameter search operation, and
        it will return you the best optimistic  model with the best parameter:
        estimator: Model object
        param_grid: dictionary of parameter to perform search operation
        input_feature: you're all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return GridSearchOperation object
        """
        try:
            # instantiating GridSearchCV class
            grid_search_cv_ref = ModelFactory.class_for_name(module_name=self.grid_search_cv_module,
                                                             class_name=self.grid_search_class_name
                                                             )

            grid_search_cv = grid_search_cv_ref(estimator=initialized_model.model,
                                                param_grid=initialized_model.param_grid_search)
            grid_search_cv = ModelFactory.update_property_of_class(grid_search_cv,
                                                                   self.grid_search_property_data)

            message = f'{">>" * 30} f"Training {type(initialized_model.model).__name__} Started." {"<<" * 30}'
            logging.info(message)
            grid_search_cv.fit(input_feature, output_feature)
            message = f'{">>" * 30} f"Training {type(initialized_model.model).__name__}" completed {"<<" * 30}'
            grid_searched_best_model = GridSearchedBestModel(model_serial_number=initialized_model.model_serial_number,
                                                             model=initialized_model.model,
                                                             best_model=grid_search_cv.best_estimator_,
                                                             best_parameters=grid_search_cv.best_params_,
                                                             best_score=grid_search_cv.best_score_
                                                             )

            return grid_searched_best_model
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def get_initialized_model_list(self) -> List[InitializedModelDetail]:
        """
        This function will return a list of model details.
        return List[ModelDetail]
        """
        try:
            initialized_model_list = []
            for model_serial_number in self.models_initialization_config.keys():

                model_initialization_config = self.models_initialization_config[model_serial_number]
                model_obj_ref = ModelFactory.class_for_name(module_name=model_initialization_config[MODULE_KEY],
                                                            class_name=model_initialization_config[CLASS_KEY]
                                                            )
                model1 = model_obj_ref()

                if PARAM_KEY in model_initialization_config:
                    model_obj_property_data = dict(model_initialization_config[PARAM_KEY])
                    model1 = ModelFactory.update_property_of_class(instance_ref=model1,
                                                                   property_data=model_obj_property_data)

                param_grid_search = model_initialization_config[SEARCH_PARAM_GRID_KEY]
                model_name = f"{model_initialization_config[MODULE_KEY]}.{model_initialization_config[CLASS_KEY]}"

                model_initialization_config = InitializedModelDetail(model_serial_number=model_serial_number,
                                                                     model=model1,
                                                                     param_grid_search=param_grid_search,
                                                                     model_name=model_name
                                                                     )

                initialized_model_list.append(model_initialization_config)

            self.initialized_model_list = initialized_model_list
            return self.initialized_model_list
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_model(self, initialized_model: InitializedModelDetail,
                                                             input_feature,
                                                             output_feature) -> GridSearchedBestModel:
        """
        initiate_best_model_parameter_search(): function will perform parameter search operation, and
        it will return you the best optimistic  model with the best parameter:
        estimator: Model object
        param_grid: dictionary of parameter to perform search operation
        input_feature: all input features
        output_feature: Target/Dependent features
        ================================================================================
        return: Function will return a GridSearchOperation
        """
        try:
            return self.execute_grid_search_operation(initialized_model=initialized_model,
                                                      input_feature=input_feature,
                                                      output_feature=output_feature)
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def initiate_best_parameter_search_for_initialized_models(self,
                                                              initialized_model_list: List[InitializedModelDetail],
                                                              input_feature,
                                                              output_feature) -> List[GridSearchedBestModel]:

        try:
            self.grid_searched_best_model_list = []
            for initialized_model_list in initialized_model_list:
                grid_searched_best_model = self.initiate_best_parameter_search_for_initialized_model(
                    initialized_model=initialized_model_list,
                    input_feature=input_feature,
                    output_feature=output_feature
                )
                self.grid_searched_best_model_list.append(grid_searched_best_model)
            return self.grid_searched_best_model_list
        except Exception as e:
            raise ApplicationException(e, sys) from e

    @staticmethod
    def get_model_detail(model_details: List[InitializedModelDetail],
                         model_serial_number: str) -> InitializedModelDetail:
        """
        This function return ModelDetail
        """
        try:
            for model_data in model_details:
                if model_data.model_serial_number == model_serial_number:
                    return model_data
        except Exception as e:
            raise ApplicationException(e, sys) from e

    @staticmethod
    def get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list: List[GridSearchedBestModel],
                                                          base_r2=0.6 # R2score
                                                          ) -> BestModel:
        try:
            best_model = None
            for grid_searched_best_model in grid_searched_best_model_list:
                if base_r2 < grid_searched_best_model.best_score:
                    
                    logging.info(f"---------------------------")
                    logging.info(f"R2_score : {grid_searched_best_model.best_score}")
                    logging.info(f"Acceptable model found:{grid_searched_best_model}")
                    base_r2 = grid_searched_best_model.best_score

                    best_model = grid_searched_best_model
            if not best_model:
                raise Exception(f"None of Model has base accuracy: {base_r2}")
            logging.info(f"Best model: {best_model}")
            return best_model
        except Exception as e:
            raise ApplicationException(e, sys) from e

    def get_best_model(self, X, y, base_r2=0.6) -> BestModel:
        try:
            logging.info("Started Initializing model from config file")
            initialized_model_list = self.get_initialized_model_list()
            logging.info(f"Initialized model: {initialized_model_list}")
            grid_searched_best_model_list = self.initiate_best_parameter_search_for_initialized_models(
                initialized_model_list=initialized_model_list,
                input_feature=X,
                output_feature=y
            )
            return ModelFactory.get_best_model_from_grid_searched_best_model_list(grid_searched_best_model_list,
                                                                                  base_r2=base_r2)
        except Exception as e:
            raise ApplicationException(e, sys)