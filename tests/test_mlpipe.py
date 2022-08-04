# pylint: skip-file
import pytest
import os
import pandas as pd

from src.app_func.mlpipe import MLpipeline


@pytest.fixture(name="fixture_MLpipeline")
def fixture_MLpipeline():
    """ Creates MLobject and teardowns any testing files associated with testing process
    Yields:
        MLpipeline (object): Instantiates MLpipeline object for testing
    """
    yield MLpipeline()
    for filename in os.listdir("output"):
        if filename.startswith("Testing"):
            os.remove(os.path.join("output", filename))


def test_model_split_X_train(fixture_MLpipeline):
    """ Testing for model split output X_train
    """
    input_data = pd.read_csv("tests/artifacts/ml_data.csv")
    X_train, _, _, _ = fixture_MLpipeline.model_split(input_data)
    expected_X_train = pd.read_csv("tests/artifacts/X_train.csv")
    assert len(X_train) == len(expected_X_train)


def test_model_split_X_test(fixture_MLpipeline):
    """ Testing for model split output X_test
    """
    input_data = pd.read_csv("tests/artifacts/ml_data.csv")
    _, X_test, _, _ = fixture_MLpipeline.model_split(input_data)
    expected_X_test = pd.read_csv("tests/artifacts/X_test.csv")
    assert len(X_test) == len(expected_X_test)


def test_model_split_y_train(fixture_MLpipeline):
    """ Testing for model split output y_train
    """
    input_data = pd.read_csv("tests/artifacts/ml_data.csv")
    _, _, y_train, _ = fixture_MLpipeline.model_split(input_data)
    expected_y_train = pd.read_csv("tests/artifacts/y_train.csv")
    assert len(y_train) == len(expected_y_train)


def test_model_split_y_test(fixture_MLpipeline):
    """ Testing for model split output y_test
    """
    input_data = pd.read_csv("tests/artifacts/ml_data.csv")
    _, _, _, y_test = fixture_MLpipeline.model_split(input_data)
    expected_y_test = pd.read_csv("tests/artifacts/y_test.csv")
    assert len(y_test) == len(expected_y_test)


def test_metrics(fixture_MLpipeline):
    """ Testing if model metrics outputs testing file with name starting with "Testing"
    """
    model = "Testing"
    y_test = pd.read_csv("tests/artifacts/y_test.csv")
    y_pred_test = pd.read_csv("tests/artifacts/y_pred_test.csv")
    y_train = pd.read_csv("tests/artifacts/y_train.csv")
    y_pred_train = pd.read_csv("tests/artifacts/y_pred_train.csv")
    fixture_MLpipeline.metrics(model, y_test, y_pred_test, y_train, y_pred_train)
    for filename in os.listdir("output"):
        if filename.startswith("Testing"):
            assert str(filename)[:7] == "Testing"
