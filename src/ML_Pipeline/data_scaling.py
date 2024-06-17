import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelBinarizer, StandardScaler, MinMaxScaler

def data_scaling(data, column_names_to_standardize):
    data_scaled = StandardScaler().fit_transform(data[column_names_to_standardize].values)
    data_temp = pd.DataFrame(data_scaled, columns=column_names_to_standardize, index = data.index)
    data[column_names_to_standardize] = data_temp
    return data