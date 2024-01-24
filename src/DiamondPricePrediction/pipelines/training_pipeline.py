from src.DiamondPricePrediction.components.data_ingestion import Data_ingestion
from src.DiamondPricePrediction.components.data_transformation import Data_transformation
from src.DiamondPricePrediction.components.model_trainer import Model_Trainer
import os
import sys
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import Customexception
import pandas as pd


obj = Data_ingestion()
train_path,test_path = obj.Initiate_Data_Ingestion()
data_transformation = Data_transformation()

train_array,test_array = data_transformation.initiate_data_transformation(train_path,test_path)

model_training = Model_Trainer()

model_training.Initiate_model_training(train_array,test_array) 

