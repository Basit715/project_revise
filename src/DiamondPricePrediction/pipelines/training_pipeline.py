from src.DiamondPricePrediction.components.data_ingestion import Data_ingestion
import os
import sys
from src.DiamondPricePrediction.logger import logging
from src.DiamondPricePrediction.exception import Customexception
import pandas as pd


obj = Data_ingestion()
obj.Initiate_Data_Ingestion()