import h2o
import numpy as np
import pandas as pd

def read_dataset_advanced(path, sort_by_col):
    data = pd.read_csv(path, low_memory=False)\
            .drop_duplicates()\
            .sort_values(by=sort_by_col)\
            .reset_index(drop=True)
    return data

def replace_outliers_with_nan(data, col, threshold_val):
        data[col] = np.where(data[col] > threshold_val, np.nan, data[col])
        return data

def club_duplicate_rows(data, id_col):
        data = data.groupby([id_col]).agg('max').reset_index()
        return data

def map_entries(data, col, mapping_dict):
        data.replace({col: mapping_dict}, inplace=True)
        return data
def replace_values(data, col, original_value, replace_with=np.nan):
        data[col]=data[col].str.replace(original_value, replace_with, regex=True)
        return data

def replace_values_with_nan(data, col, original_value):
        data[col]=data[col].replace(original_value, np.nan, regex=True)
        return data