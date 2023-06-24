import pandas as pd
import json
import logging
from Concrete_Strength_Prediction.constant import *
import pymongo
import yaml
from Concrete_Strength_Prediction.data_access.mongo_db_connection import MongoDBClient


class prediction_upload:
    def __init__(self, DATABASE_NAME_PREDICTION, COLLECTION_NAME_PREDICTION):
        self.data_base = DATABASE_NAME_PREDICTION
        self.collection_name = COLLECTION_NAME_PREDICTION
        self.mongo_client = MongoDBClient(database_name=DATABASE_NAME_PREDICTION)
        
    def data_dump(self, filepath):
        df = pd.read_csv(filepath)
        print(f"Rows and columns: {df.shape}")

        # Convert dataframe to json so that we can dump these records into MongoDB
        df.reset_index(drop=True, inplace=True)
        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        json_record = list(json.loads(df.T.to_json()).values())
        print(json_record[0])

        print("Data Uploaded")

        # Check if the database exists
        database_names = self.mongo_client.client.list_database_names()
        if self.data_base in database_names:
            print(f"The database {self.data_base} already exists")
            # Check if the collection exists
            if self.collection_name in self.mongo_client.database.list_collection_names():
                print(f"The collection {self.collection_name} already exists")
                # Drop the existing collection
                self.mongo_client.database[self.collection_name].drop()
                print(f"The collection {self.collection_name} is dropped and will be replaced with new data")
            else:
                print(f"The collection {self.collection_name} does not exist and will be created")
        else:
            # Create the database and collection
            print(f"The database {self.data_base} does not exist and will be created")
            db = self.mongo_client.client[self.data_base]
            col = db[self.collection_name]
            print(f"The collection {self.collection_name} is created")

        # Insert converted json record into MongoDB
        self.mongo_client.database[self.collection_name].insert_many(json_record)

        logging.info("Prediction Data Updated to MongoDB")
