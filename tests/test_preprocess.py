# pylint: skip-file
import pytest
import numpy as np
import pandas as pd

from src.preprocess import Preprocesor


@pytest.fixture(name="fixture_Preprocess")
def fixture_Preprocess():
    """Pytest fixture for Preprocess Object
    Returns:
        fixture_extract (object): returns ImportData object
    """
    yield Preprocesor()


def test_preprocess_df(fixture_Preprocess):
    """ Testing to see if dataframe output is of the same shape as expected 
    """
    output = fixture_Preprocess.preprocess_df()
    expected = pd.read_csv("tests/artifacts/processed_df.csv")
    assert expected.shape == output.shape


def test__split_price(fixture_Preprocess):
    """ Testing to see if price values are output according to correct format
    """
    input_price = ["None", None, "SGD$ 14.55", "USD$ 14.00"]
    expected = [np.nan, np.nan, float(14.55), float(14)]
    for ind, val in enumerate(input_price):
        if val in ("None", None):
            assert np.isnan(expected[ind]) and np.isnan(
                fixture_Preprocess._split_price(val)
            )
        else:
            assert expected[ind] == fixture_Preprocess._split_price(val)


def test__chg_to_num(fixture_Preprocess):
    input_num = [1, 3, "one", "four"]
    expected = [1, 3, 1, 4]
    for ind, val in enumerate(input_num):
        assert expected[ind] == fixture_Preprocess._chg_to_num(val)


def test_to_bool(fixture_Preprocess):
    input_non_bool = ["Yes", "No", None, np.nan]
    expected = [1, 0, 0, 0]
    for ind, val in enumerate(input_non_bool):
        assert expected[ind] == fixture_Preprocess._to_bool(val)


def test_checkout_neg(fixture_Preprocess):
    input_checkout_neg = [-1, -5, 7, 1000]
    expected = [1, 5, 7, 1000]
    for ind, val in enumerate(input_checkout_neg):
        assert expected[ind] == fixture_Preprocess._checkout_neg(val)


def test_title_case(fixture_Preprocess):
    input_title_case = ["April", "ApRiL", "SeptemBer", "OctobeR"]
    expected = ["April", "April", "September", "October"]
    for ind, val in enumerate(input_title_case):
        assert expected[ind] == fixture_Preprocess._title_case(val)
