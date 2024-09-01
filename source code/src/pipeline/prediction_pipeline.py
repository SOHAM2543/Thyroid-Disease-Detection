import sys,os
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import pandas as pd, numpy as np

class PredictPipeline:
    def predict(self,features):
        try:
            preprocessor_path = os.path.join('artifacts','preprocessor.pkl')
            model_path = os.path.join('artifacts','model.pkl')

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Exception occured in prediction")
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,age,on_thyroxine,goitre,TSH,TSH_measured,T3,TT4_measured,TT4,FTI,referral_source):
        self.age = age
        self.on_thyroxine = on_thyroxine
        self.goitre = goitre
        self.TSH = TSH
        self.TSH_measured = TSH_measured
        self.T3 = T3
        self.TT4_measured = TT4_measured
        self.TT4 = TT4
        self.FTI = FTI
        self.referral_source = referral_source

    def get_data_as_dataframe(self):
        try:
            
            custom_data_input_dict = {
                "age" : [self.age],"on_thyroxine":[self.on_thyroxine],"goitre":[self.goitre],"TSH":[self.TSH],
                "TSH_measured":[self.TSH_measured],"T3":[self.T3],"TT4_measured":[self.TT4_measured],"TT4":[self.TT4],
                "FTI":[self.FTI],"referral_source":[self.referral_source]
            }

            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            return df

        except Exception as e:
            logging.error('Exception Occurred in prediction pipeline')
            raise CustomException(e, sys)