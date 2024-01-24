import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

from dataclasses import dataclass
from src.DiamondPricePrediction.exception import Customexception
from src.DiamondPricePrediction.logger import logging
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from src.DiamondPricePrediction.utils.utils import save_object

@dataclass
class DataTransformationConfig:
     preproccesor_obj_file = os.path.join('artifacts', 'preproccesor.pkl')

class Data_transformation:
     def __init__(self):
          self.data_transformation_config = DataTransformationConfig()
     
     def get_data_transformation(self):
          try:
               categorical_cols = ['cut', 'color', 'clarity']
               numerical_cols = ['carat', 'depth', 'table', 'x','y','z']
               
               
               cut_categories = ['Fair','Good','Very Good', 'Premium', 'Ideal']
               clarity_categories = ['I1', 'SI2', 'SI1', 'VS2', 'VS1','VVS1', 'VVS2', 'VVS1', 'IF']
               color_categories = ['D', 'E', 'F', 'G', 'H', 'I', 'J']
               
               
               num_pipeline = Pipeline(
                    steps=[
                              ('imputer', SimpleImputer()),
                              ('scaler', StandardScaler()),
         
          
                                   ]
                    )
               
               cat_pipeline = Pipeline(
                    steps=[
                         ('imputer', SimpleImputer(strategy='most_frequent')),
                         ('encoder', OrdinalEncoder(categories=[cut_categories, color_categories, clarity_categories]))
                         ]
                    )
               
               preproccesor = ColumnTransformer(
     [
          ('num pipeline', num_pipeline, numerical_cols),
          ('cat pipeline',cat_pipeline, categorical_cols)
          
     ]
)
               
               
               return preproccesor
          except Exception as e:
               
               
               logging.info('exception occured at data transformation')
               raise Customexception(e,sys)
          
     def initiate_data_transformation(self, train_path, test_path):
          try:
               train_df = pd.read_csv(train_path)
               test_df = pd.read_csv(test_path)
               logging.info('read train and test data completed')
               logging.info(f'train data head: \n{train_df.head().to_string()}')
               logging.info(f'test data head: \n{test_df.head().to_string()}')
               
               preprocessing_obj = self.get_data_transformation()
               target_coloumn = 'price'
               drop_coloumns = [target_coloumn,'id']
               
               input_feature_train_df  = train_df.drop(columns=drop_coloumns, axis=1)
               input_feature_test_df = test_df.drop(columns=drop_coloumns, axis=1)
               target_feature_train_df = train_df[target_coloumn]
               target_feature_test_df = test_df[target_coloumn]
               
               input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
               input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)
               
               train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
               test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]
               
               
               save_object(
                    file_path = self.data_transformation_config.preproccesor_obj_file,
                    obj = preprocessing_obj
               )
               logging.info('preproccessor pickle file saved')
               return(
                    train_arr,
                    test_arr
               )
                
               
          except Exception as e:
               pass
     
     