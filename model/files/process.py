import pandas as pd
import os


def make():
    module_dir = os.path.dirname(__file__)
    file_path = os.path.join(module_dir, 'data.csv')
    f = open(file_path)
    data = pd.read_csv(f)

    return data