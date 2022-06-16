import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix

from config.config_load import read_yaml_file
from preprocess import Preprocess

class MLpipeline:
    """Class that wraps the machine learning pipeline
    """
    def __init__(self):
        """Instantiate MLpipeline object and load data
        """
        config = read_yaml_file()
        self.dependent_y = config["mlpipeline"]["dependent_y"]
        self.drop = config["mlpipeline"]["drop"]

    def logregclassifier_pipeline(self, data):
        """Model Pipeline for Logistic Regression Model
        Args:
            data (dataframe): Data as created after the preprocessing steps
        """
        data.drop(labels=self.drop, axis=1, inplace=True)
        X_train, X_test, y_train, y_test = MLpipeline().model_split(data)
        ct = ColumnTransformer([
                ("OHE", OneHotEncoder(), ["branch", "country", "first_time"]), 
                ("SI", SimpleImputer(strategy="mean"), ["SGD_price"])
                ], remainder='drop')
        pipe = Pipeline([
            ("column_transfomer", ct),
            ("logregclassifier", LogisticRegression(penalty="l1", solver='liblinear', max_iter=100))
            ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        MLpipeline().metrics("Logistic Regression", y_test, y_pred)

    # def decisiontreeclassifier():
    #     pass

    # def randomforestclassifier():
    #     pass

    def model_split(self, data):
        """Splits the data into training and testing sets
        Args:
            data (dataframe): dataframe to be put into machine learning model
        Returns:
            X_train: independent variable used to train the model
            X_test: dependent variable used to evaluate trained model
            y_train: independent variable used for prediction
            y_test: dependent variable used to evaluate prediction
        """
        X = data.drop(columns=self.dependent_y)
        y = data[self.dependent_y]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
        return X_train, X_test, y_train, y_test

    def metrics(self, model, y_true, y_pred):
        """Prints out the model and various classifier metrics to compare model performance
        Args:
            model (string): Name of model executed
            y_true (series): Series of actual test values by the model
            y_pred (series): Series of predicted values by the model
        """
        date_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        accuracy = ((tp+tn) / (tn+fp+fn+tp))*100
        precision = (tp/(tp+fp))*100
        recall = (tp/(tp+fn))*100
        f1_score = ((2*precision*recall)/(precision+recall))
        print(f"Model Evaluation Metrics for {model} at time {date_time}")
        print(f"Model Accuracy {accuracy:.2f}%")
        print(f"Model F1-Score {f1_score:.2f}%")
        print(f"Model Precision {precision:.2f}%")
        print(f"Model Recall {recall:.2f}%")

if __name__=="__main__":
    df = Preprocess().preprocess_df()
    MLpipeline().logregclassifier_pipeline(df)