import numpy as np
from pymongo import MongoClient
import pandas as pd
import json
import logging
from src.logger import logging

# Inserting the data into Database
class DBConn:
 client = MongoClient("mongodb+srv://scoot:root@atlascluster.onql57n.mongodb.net/?retryWrites=true&w=majority")

 def insert_data(self, collected_data):
     db = self.client['IneuronIntersip']
     collection = db['ThyroidData']

    #  # Convert DataFrame to dictionary (records format is often most useful)
    #  data_dict = collected_data.to_dict(orient='records')

     # Insert the data into MongoDB
     collection.insert_one(collected_data)
     logging.debug('All the data has been exported to MongoDB server')
