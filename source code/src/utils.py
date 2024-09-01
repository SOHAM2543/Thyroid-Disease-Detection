import os,sys,pickle
import numpy as np
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from sklearn.metrics import accuracy_score,classification_report

def save_object(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        for model_name, model in models.items():
            # Train model
            model.fit(X_train, y_train)

            # Predict test data
            y_pred = model.predict(X_test)

            # Accuracy Score
            accuracy = accuracy_score(y_test, y_pred)

            # Classification Report
            class_report = classification_report(y_test, y_pred, output_dict=True)

            logging.info(f'{model_name} Accuracy Score: {accuracy}')
            logging.info(f'{model_name} Classification Report: \n{classification_report(y_test, y_pred)}')

            # Storing the accuracy score to compare models
            report[model_name] = accuracy

        return report
    
    except Exception as e:
        logging.info('Exception occurred during model evaluation')
        raise CustomException(e, sys)
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception occurred in load_object function')
        raise CustomException(e, sys)