import pandas as pd

def load_data(file_path):
    """
    Loads data from a given CSV file.
    :param file_path: Path to the CSV file.
    :return: A Pandas DataFrame containing the data.
    """
    return pd.read_csv(file_path)