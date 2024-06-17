import numpy as np
import pandas as pd

def categorical_encoding(data, col, prefix):
    dummies = pd.get_dummies(data[col], prefix=prefix)
    data[dummies.columns] = dummies
    data = data.drop([col], axis=1)
    return data