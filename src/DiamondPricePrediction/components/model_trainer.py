import pandas as pd
import numpy as np
import os
import sys
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import Customexception
from dataclasses import dataclass
from src.DiamondPricePrediction.utils.utils import save_object
from src.DiamondPricePrediction.utils.utils import evaluate_model

from sklearn.linear_model import LinearRegression, Lasso,Ridge,ElasticNet

@dataclass
class ModelTrainingConfig:
     trained_model_file_path = os.path.join('artifacts', 'pkl')
     
     



class Model_Trainer:
     def __init__(self):
          self.model_trainer_config = ModelTrainingConfig()
     
     def Initiate_model_training(self,train_array,test_array):
          try:
               logging.info('splitting independant and dependant variables from train and test data')
               x_train,y_train,x_test,y_test = (
                    train_array[:, :-1],
                    train_array[:,-1],
                    test_array[:,:-1],
                    test_array[:,-1]
               )
               
               models = {
                    'LinearRegression': LinearRegression(),
                    'Ridge': Ridge(),
                    'Lasso':Lasso(),
                    'ElasticNet':ElasticNet()
}
               
               model_report:dict = evaluate_model(x_train,y_train,x_test,y_test,models)
               print(model_report)
               print('\n====================================================\n')
               logging.info(f'model report: {model_report}')
               best_model_score = max(sorted(model_report.values()))
               best_model_name = list(model_report.keys())[
                    list(model_report.values()).index(best_model_score) 
               ] 
               best_model = models[best_model_name]
               print(f'Best model found, model name : {best_model_name}, R2 score : {best_model_score}')
               print('\n=================================\n')
               logging.info(f'Best model found, model name : {best_model_name}, R2 score : {best_model_score}') 
               save_object(
                    file_path = self.model_trainer_config.trained_model_file_path,
                    obj = best_model
               ) 
               
               
               
               
          except Exception as e:
               logging.info('exception occured at model training')
               raise Customexception(e,sys) 
     
     
     