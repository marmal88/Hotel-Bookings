# pylint: skip-file
import pytest
import numpy as np

from src.preprocess import Preprocess
from src.extract import ImportData
from src.config.config_load import read_test_yaml_file


@pytest.fixture(name="fixture_preprocess")
def fixture_preprocess():
    """Pytest fixture for Preprocess Object
    Returns:
        fixture_extract (object): returns ImportData object
    """
    yield Preprocess()

def test_preprocess_df(fixture_preprocess):
    pass

def test__split_price(fixture_preprocess):
    pass
    # input_price = ["None", np.nan, "SGD$ 75.25", "USD$ 43.00"]
    # expected = [np.nan, np.nan, float(75.25), float(43.25)]
    # assert fixture_preprocess()._split_price(input_price) == expected
