import datetime
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report

from src.config.config_load import read_yaml_file


class MLpipeline:
    """Class that wraps the machine learning pipeline
    """

    def __init__(self):
        """Instantiate MLpipeline object and load data
        """
        self.config = read_yaml_file()
        self.dependent_y = self.config["mlpipeline"]["dependent_y"]
        self.drop = self.config["mlpipeline"]["drop"]
        self.ohe1 = self.config["feature_eng"]["ohe1"]
        self.ore1 = self.config["feature_eng"]["ore1"]
        self.sim1 = self.config["feature_eng"]["sim1"]
        self.ore2 = self.config["feature_eng"]["ore2"]
        self.sim2 = self.config["feature_eng"]["sim2"]
        self.bins = self.config["preprocess"]["bins"]
        self.encode = self.config["preprocess"]["encode"]
        self.strategy_bin = self.config["preprocess"]["strategy"]
        self.strategy_si = self.config["feature_eng"]["strategy"]

    def ml_workflow(self, data, model):
        """Model Pipeline for Logistic Regression Model
        Args:
            data (dataframe): Data as created after the preprocessing steps
            model (string): Name of model to use
        """
        if isinstance(model, str):
            if model == "lr":
                model_name = self.config["mlpipeline"]["lr"]["name"]
                penalty = self.config["mlpipeline"]["lr"]["params"]["penalty"]
                solver = self.config["mlpipeline"]["lr"]["params"]["solver"]
                max_iter = self.config["mlpipeline"]["lr"]["params"]["max_iter"]
                c_strength = self.config["mlpipeline"]["lr"]["params"]["c"]
                model_obj = LogisticRegression(
                    penalty=penalty,
                    solver=solver,
                    max_iter=max_iter,
                    C=c_strength,
                    random_state=11,
                )
            elif model == "dt":
                model_name = self.config["mlpipeline"]["dt"]["name"]
                max_depth = self.config["mlpipeline"]["dt"]["params"]["max_depth"]
                min_samples_split = self.config["mlpipeline"]["dt"]["params"][
                    "min_samples_split"
                ]
                min_samples_leaf = self.config["mlpipeline"]["dt"]["params"][
                    "min_samples_leaf"
                ]
                criterion = self.config["mlpipeline"]["dt"]["params"]["criterion"]
                splitter = self.config["mlpipeline"]["dt"]["params"]["splitter"]
                model_obj = DecisionTreeClassifier(
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    criterion=criterion,
                    splitter=splitter,
                    random_state=11,
                )
            elif model == "rf":
                model_name = self.config["mlpipeline"]["rf"]["name"]
                n_estimators = self.config["mlpipeline"]["rf"]["params"]["n_estimators"]
                criterion = self.config["mlpipeline"]["rf"]["params"]["criterion"]
                max_depth = self.config["mlpipeline"]["rf"]["params"]["max_depth"]
                min_samples_split = self.config["mlpipeline"]["rf"]["params"][
                    "min_samples_split"
                ]
                min_samples_leaf = self.config["mlpipeline"]["rf"]["params"][
                    "min_samples_leaf"
                ]
                max_leaf_nodes = self.config["mlpipeline"]["rf"]["params"][
                    "max_leaf_nodes"
                ]
                max_features = self.config["mlpipeline"]["rf"]["params"]["max_features"]
                if not max_features:
                    max_features = None
                model_obj = RandomForestClassifier(
                    n_estimators=n_estimators,
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    min_samples_leaf=min_samples_leaf,
                    max_leaf_nodes=max_leaf_nodes,
                    max_features=max_features,
                    random_state=11,
                )
            else:
                print("Invalid model, please check if model exists in config")
        ml_data = data.drop(labels=self.drop, axis=1)
        X_train, X_test, y_train, y_test = MLpipeline().model_split(ml_data)
        price_transformer = Pipeline(
            [
                ("sim1", SimpleImputer(strategy=self.strategy_si)),
                (
                    "bin1",
                    KBinsDiscretizer(
                        n_bins=self.bins, encode=self.encode, strategy=self.strategy_bin
                    ),
                ),
            ]
        )
        if model == "lr":
            ct = ColumnTransformer(
                [
                    ("ohe1", OneHotEncoder(drop="first"), self.ohe1),
                    ("ore1", OrdinalEncoder(encoded_missing_value=-1), self.ore1),
                    ("num1", price_transformer, self.sim1),
                ],
                remainder="drop",
            )
        else:
            ct = ColumnTransformer(
                [
                    ("ore2", OrdinalEncoder(encoded_missing_value=-1), self.ore2),
                    ("num1", price_transformer, self.sim1),
                ],
                remainder="passthrough",
            )
        pipe = Pipeline([("column_transfomer", ct), ("classifier", model_obj)])
        pipe.fit(X_train, y_train)
        y_pred_train = pipe.predict(X_train)
        y_pred_test = pipe.predict(X_test)
        MLpipeline().metrics(model_name, y_test, y_pred_test, y_train, y_pred_train)

        return

    def model_split(self, data):
        """Splits the data into training and testing sets
        Args:
            data (dataframe): dataframe to be put into machine learning model
        Returns:
            X_train (dataframe): independent variable used to train the model
            X_test (series): dependent variable used to evaluate trained model
            y_train (dataframe): independent variable used for prediction
            y_test (series): dependent variable used to evaluate prediction
        """
        X = data.drop(columns=self.dependent_y)
        y = data[self.dependent_y]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=11
        )
        return X_train, X_test, y_train, y_test

    def metrics(self, model, y_test, y_pred_test, y_train, y_pred_train):
        """Prints out the model and various classifier metrics to compare model performance
        Args:
            model (string): Name of model executed
            y_test (series): Series of actual test values by the model
            y_pred_test (series): Series of predicted values by the model
            y_train (series): Series of actual train values by the model
            y_pred_train (series): Series of predicted train values by the model
        """
        results = self.config["mlpipeline"]["results"]
        classreport = self.config["mlpipeline"]["class_report"]
        date_time = datetime.datetime.now().strftime("%Y-%m-%d-H%H-M%M-S%S")
        # date_time_file = datetime.datetime.now().strftime("%Y-%m-%d")
        directory = "output"
        filename = model + "_Evaluation_Metrics_" + date_time + ".txt"
        with open(os.path.join(directory, filename), "w", encoding="utf-8") as files:
            files.write(f"\nModel Evaluation Metrics for {model} at time {date_time}")
            if results:
                tr_tn, tr_fp, tr_fn, tr_tp = confusion_matrix(
                    y_train, y_pred_train
                ).ravel()
                train_acc = ((tr_tp + tr_tn) / (tr_tn + tr_fp + tr_fn + tr_tp)) * 100
                train_prec = (tr_tp / (tr_tp + tr_fp)) * 100
                train_recall = (tr_tp / (tr_tp + tr_fn)) * 100
                train_f1 = (2 * train_prec * train_recall) / (train_prec + train_recall)
                files.write(f"\n{model} Train Set Model Metrics")
                files.write(f"\nTrain Set Precision {train_prec:.2f}%")
                files.write(f"\nTrain Set Accuracy {train_acc:.2f}%")
                files.write(f"\nTrain Set Recall {train_recall:.2f}%")
                files.write(f"\nTrain Set F1-Score {train_f1:.2f}%")
            if classreport:
                files.write(f"\n{classification_report(y_train, y_pred_train)}")

            t_tn, t_fp, t_fn, t_tp = confusion_matrix(y_test, y_pred_test).ravel()
            test_acc = ((t_tp + t_tn) / (t_tn + t_fp + t_fn + t_tp)) * 100
            test_prec = (t_tp / (t_tp + t_fp)) * 100
            test_recall = (t_tp / (t_tp + t_fn)) * 100
            test_f1 = (2 * test_prec * test_recall) / (test_prec + test_recall)
            files.write(f"\n{model} Test Set Model Metrics")
            files.write(f"\nTest Set Precision {test_prec:.2f}%")
            files.write(f"\nTest Set Accuracy {test_acc:.2f}%")
            files.write(f"\nTest Set Recall {test_recall:.2f}%")
            files.write(f"\nTest Set F1-Score {test_f1:.2f}%")

            if classreport:
                files.write(f"\n{classification_report(y_test, y_pred_test)}")

    def frontend_output(self, data, X_test):

        model_name = self.config["mlpipeline"]["dt"]["name"]
        max_depth = self.config["mlpipeline"]["dt"]["params"]["max_depth"]
        min_samples_split = self.config["mlpipeline"]["dt"]["params"][
            "min_samples_split"
        ]
        min_samples_leaf = self.config["mlpipeline"]["dt"]["params"]["min_samples_leaf"]
        criterion = self.config["mlpipeline"]["dt"]["params"]["criterion"]
        splitter = self.config["mlpipeline"]["dt"]["params"]["splitter"]
        model_obj = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            criterion=criterion,
            splitter=splitter,
            random_state=11,
        )
        X_train, _, y_train, _ = MLpipeline().model_split(data)
        price_transformer = Pipeline(
            [
                ("sim1", SimpleImputer(strategy=self.strategy_si)),
                (
                    "bin1",
                    KBinsDiscretizer(
                        n_bins=self.bins, encode=self.encode, strategy=self.strategy_bin
                    ),
                ),
            ]
        )
        ct = ColumnTransformer(
            [
                ("ore2", OrdinalEncoder(encoded_missing_value=-1), self.ore2),
                ("num1", price_transformer, self.sim1),
            ],
            remainder="passthrough",
        )
        pipe_final = Pipeline([("column_transfomer", ct), ("classifier", model_obj)])
        pipe_final.fit(X_train, y_train)
        y_pred_prob = pipe_final.predict_proba(X_test)
        y_pred_test = pipe_final.predict(X_test)
        y_pred_test = y_pred_test[0]
        if y_pred_test:
            y_pred_prob = y_pred_prob[0][1]
        else:
            y_pred_prob = y_pred_prob[0][0]

        return model_name, y_pred_prob, y_pred_test
