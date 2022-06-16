import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from config.config_load import read_yaml_file
from extract import ImportData
from preprocess import Preprocess

class MLpipeline:
    """Class that wraps the machine learning pipeline
    """
    def __init__(self, data):
        config = read_yaml_file()
        self.dependent_y = config["mlpipeline"]["dependent_y"]
        self.data = data
        print(config["preprocess"]["int_cols"])

    # def mlpipe(self, data):
    #     ct = ColumnTransformer([
    #         ("step1", StandardScaler(), []),
    #         (),
    #         remainer=drop
    #         ])
    #     for each in 
    #     p = Pipeline([
    #         ("column_transfomer", ct),
    #         ("model", )
    #         ])
    #     p.fit(X_train, y_train)
    #     p.predict(X_test)

    # def model_split(self):
    #     X = self.data.drop(columns=self.dependent_y)
    #     y = self.data[self.dependent_y]
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

    # def logregclassifier():
    #     pass

    # def randomforestclassifier():
    #     pass

    # def decisiontreeclassifier():
    #     pass

    # def metrics(self):
    #     pass

if __name__=="__main__":
    df = Preprocess().preprocess_df()
    MLpipeline(df)