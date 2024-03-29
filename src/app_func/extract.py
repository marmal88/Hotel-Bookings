import pandas as pd
from sqlalchemy import create_engine


class ImportData:
    """Class to encapsulate sql query responses, parse and return dataframes
    """

    def __init__(self, data_location):
        """instantiate object to connect to sql engine
        Args:
            data_location (string): location of the data.db file
        """
        self.data_location = data_location
        engine = create_engine("sqlite:///" + self.data_location)
        self.conn = engine.connect()

    def return_table(self, table):
        """Returns the datasets found in the SQL connection
        Args:
            table (string): name of table to query
        Returns:
            articles (dataframe) : Returns dataframe with from articles table
        """
        if isinstance(table, str):
            query = "SELECT * FROM " + table
            query_table = self.conn.execute(query).all()
            query_table_keys = list(self.conn.execute(query).keys())
            dataframe_tbl = pd.DataFrame(query_table, columns=query_table_keys)
            return dataframe_tbl
        print("This is not a valid table")
