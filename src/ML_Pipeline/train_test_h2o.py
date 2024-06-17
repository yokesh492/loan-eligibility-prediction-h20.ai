import pandas as pd 
import numpy as np 
import h2o

def train_test_h2o(df, ratios = [.8]):
    hf = h2o.H2OFrame(df)
    train, test = hf.split_frame(ratios = [.8], seed = 1234)
    return train, test