import pandas as pd
import numpy as np

from src.extract import ImportData
from src.config.config_load import read_yaml_file


class Preprocess:
    """Class that wraps the preprocesing pipeline before handover to Machine Learning Pipeline
    """

    def __init__(self):
        """Instantiate preprocessing object and load data
        Args:
            config (yaml): config yaml file
        """
        self.config = read_yaml_file()
        self.data_location = self.config["data"]["data_location"]
        self.data_table = self.config["data"]["data_table"]
        self.int_cols = self.config["preprocess"]["int_cols"]
        self.exchange = self.config["preprocess"]["exchange"]
        self.bins = self.config["preprocess"]["bins"]
        self.labels = self.config["preprocess"]["labels"]
        self.data = ImportData(self.data_location).return_table(self.data_table)

    def preprocess_df(self):
        """Main pipeline for preprocessing steps
        Returns:
            df: data frame after preprocessing steps taken
        """
        if self.data is not None:
            self.data = self.data.set_index("booking_id")
            self.data.dropna(how="all", axis=0, inplace=True)
            self.data = self.data.replace(["None", "nan"], np.nan)
            self.data["currency"] = self.data.price.apply(
                lambda x: np.nan if x is None else x[:3]
            ).astype("category")
            self.data["price"] = self.data.price.apply(self._split_price)
            self.data["SGD_price"] = np.where(
                self.data["currency"] == "USD",
                round(self.data["price"] * self.exchange, 2),
                self.data["price"],
            )
            self.data["num_adults"] = self.data.num_adults.apply(self._chg_to_num)
            self.data[self.int_cols] = self.data[self.int_cols].astype("int")
            self.data["first_time"] = self.data["first_time"].apply(self._to_bool)
            self.data["no_show"] = self.data["no_show"].apply(self._to_bool)
            self.data["checkout_day"] = self.data.checkout_day.apply(self._checkout_neg)
            self.data["arrival_month"] = (
                self.data["arrival_month"].apply(self._title_case).astype("category")
            )
            self.data["price_types"] = pd.cut(
                self.data.SGD_price, bins=self.bins, labels=self.labels, ordered=True
            )
            return self.data.reset_index()
        print("Please check if data exists in config location")

    def _split_price(self, price):
        """Parse out the relevant price from the series
        Args:
            price (string): price with full currency and price
        Returns:
            price: float point price
        """
        if price == "None" or price is None:
            return np.nan
        return float(price[5:])

    def _chg_to_num(self, num):
        """Change numbers from string to integers for use in lambda only
        Args:
            num (string): string version of integer
        Returns:
            num: numerical version of string integer
        """
        if num is None:
            return np.nan
        elif num == "one":
            return int(1)
        elif num == "two":
            return int(2)
        else:
            return int(num)

    def _to_bool(self, non_bool):
        """Change from Non Boolean to Boolean values
        Args:
            non_bool: can either be 1 or "Yes"
        Returns:
            non_bool: boolean version of string
        """
        if non_bool in ("Yes", 1):
            return True
        return False

    def _checkout_neg(self, num):
        """For each day provided return the absolute number
        Args:
            num (integer): checkout day provided
        Returns:
            num: absolute value of number given
        """
        if num < 0:
            return abs(num)
        return num

    def _title_case(self, month):
        """Change input into titlecase
        Args:
            month (string): month input
        Returns:
            month: month in title case
        """
        if isinstance(month, str):
            return month.title()
        return month
