import sys,os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformation_object1(self):
        try:
            
            logging.info("Data Transformation Initiate")
            on_thyroxine_categories = ['f','t']
            goitre_categories = ['f','t']
            TSH_measured_categories = ['f','t']
            TT4_measured_categories = ['f','t']
            referral_source_categories = ['SVHC','other','SVI','STMW','SVHD']

            # Define numerical columns and categorical features with ordinal encoding
            numerical_cols = ['age', 'TSH', 'T3', 'TT4', 'FTI']
            ordinal_encoded_features = ['on_thyroxine', 'goitre', 'TSH_measured', 'TT4_measured', 'referral_source']
            ordinal_categories = [on_thyroxine_categories, goitre_categories, TSH_measured_categories, TT4_measured_categories, referral_source_categories]

            logging.info("Pipeline Initiated")

            # Numerical pipeline
            num_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            # Ordinal Encoded Pipeline
            ordinal_encoded_pipeline = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ordinal_encoder', OrdinalEncoder(categories=ordinal_categories)),
                ('scaler', StandardScaler())
            ])

            # Preprocessor with ColumnTransformer
            preprocessor = ColumnTransformer(transformers=[
                ('num_pipeline', num_pipeline, numerical_cols),
                ('ordinal_encoded_pipeline', ordinal_encoded_pipeline, ordinal_encoded_features)
            ])

            logging.info('Pipeline Completed')
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            # Reading train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Read train and test data completed')
            logging.info(f'Train Dataframe Head : \n{train_df.head().to_string()}')
            logging.info(f'Test Dataframe Head  : \n{test_df.head().to_string()}')

            logging.info('Obtaining preprocessing object')
            preprocessing_obj = self.get_data_transformation_object1()
            logging.info('preprocessing object returned')

            target_column_name = 'classes'
            drop_columns = [target_column_name]

            # # Check if columns exist before dropping them
            # for col in drop_columns:
            #     if col not in train_df.columns:
            #         logging.warning(f"Column {col} not found in training data. Skipping drop.")
            #         drop_columns.remove(col)
                    
            #     if col not in test_df.columns:
            #         logging.warning(f"Column {col} not found in test data. Skipping drop.")
            #         drop_columns.remove(col)

            input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
            target_feature_test_df = test_df[target_column_name]

            ## Trnasformating using preprocessor obj
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Applying preprocessing object on training and testing datasets.")
            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            # Logging the head of transformed datasets
            logging.info(f'Transformed Train Data (Head): \n{pd.DataFrame(train_arr).head().to_string()}')
            logging.info(f'Transformed Test Data (Head): \n{pd.DataFrame(test_arr).head().to_string()}')

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info('Preprocessor pickle file saved')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_datatransformation")
            raise CustomException(e, sys)