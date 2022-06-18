import pandas as pd
import pytest

# pylint: skip-file
from src.config.config_load import read_yaml_file
from src.extract import ImportData


@pytest.fixture(name="fixture_extract")
def fixture_extract():
    """Pytest fixture for ImportData Object
    Returns:
        fixture_extract (object): returns ImportData object
    """
    config = read_yaml_file()
    data_location = config["data"]["data_location"]
    yield ImportData(data_location)


def test_dataframe(fixture_extract):
    """Testing to see if return is indeed dataframe object type
    Args:
        fixture_extract (object): ImportData object
    """
    config = read_yaml_file()
    data_table = config["data"]["data_table"]
    assert isinstance(fixture_extract.return_table(data_table), pd.DataFrame)
