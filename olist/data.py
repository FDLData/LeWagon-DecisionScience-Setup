import os
import pandas as pd


class Olist:
    def get_data(self):
        """
        This function returns a Python dict.
        Its keys should be 'sellers', 'orders', 'order_items' etc...
        Its values should be pandas.DataFrames loaded from csv files
        """
        # Create the absolute path
        root_dir = os.path.dirname(os.path.dirname(__file__))
        csv_path = os.path.join(root_dir, "data", "csv")

        # Create the keys of the dictionnary
        file_names = os.listdir(csv_path)
        file_names.remove('.keep')

        file_names_cleared = file_names.copy()
        for i,file in enumerate(file_names):
            file_names_cleared[i] = file_names_cleared[i].replace(".csv", "")
            file_names_cleared[i] = file_names_cleared[i].replace("_dataset", "")
            file_names_cleared[i] = file_names_cleared[i].replace("olist_", "")

        # Create the dictionnary of data
        data = {}

        for (k, v) in zip(file_names_cleared, file_names):
            data[k] = pd.read_csv(os.path.join(csv_path, v))

        return data


    def ping(self):
        """
        You call ping I print pong.
        """
        print("pong")
