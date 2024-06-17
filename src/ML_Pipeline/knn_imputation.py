from sklearn.impute import KNNImputer
import pandas as pd 
import numpy as np 

def knn_imputation(data, column_names_to_impute, n_neighbors=5):
    imputer = KNNImputer(n_neighbors=n_neighbors)
    imputed = imputer.fit_transform(data[column_names_to_impute].values)
    data_temp = pd.DataFrame(imputed, columns=column_names_to_impute, index = data.index)
    data[column_names_to_impute] = data_temp
    return data