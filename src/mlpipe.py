import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config.config_load import read_yaml_file
from src.extract import ImportData

class MLpipeline:
    """_summary_
    """
    def __init__(self, data):
        self.data = data

    def splitdata(self):
        pass

    def Preprocess(self):
        pass