import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from src.logger import logging
from src.exception import CustomException
from src.utils import save_object, evaluate_model
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initate_model_training(self, train_array, test_array):
        try:
            logging.info('Splitting Dependent and Independent variables from train and test data')
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            models = {
                'Logistic_Regression': LogisticRegression(),
                'Decision_Tree': DecisionTreeClassifier(max_depth=3, min_samples_leaf=2, min_samples_split=3)
            }

            # Evaluate models
            model_report = evaluate_model(X_train, y_train, X_test, y_test, models)

            # Get the best model
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)
            best_model = models[best_model_name]

            logging.info(f'Best Model Found: {best_model_name} with a score of {best_model_score}')
            
            # Save the best model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            print(f'Best Model Found: {best_model_name} with a score of {best_model_score}')
            print('\n====================================================================================\n')
            
        except Exception as e:
            logging.info('Exception occurred during model training')
            raise CustomException(e, sys)
