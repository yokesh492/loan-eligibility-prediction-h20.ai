import pandas as pd 
import numpy as np
import h2o
from h2o.estimators import H2OXGBoostEstimator

def xgb_model_h2o(train_data, predictors_col_names, response_col_name, test_data, model_save_path='output', ntrees=200):
    '''
    Note : This function might not work on Windows OS
    '''
    xgb = H2OXGBoostEstimator(ntrees = ntrees, seed=1234)
    xgb.train(x=predictors_col_names, y=response_col_name, training_frame=train_data)
    perf = xgb.model_performance(test_data)
    model_path = h2o.save_model(
            xgb,
            path = model_save_path,
            force = True,
            export_cross_validation_predictions = False)
    print(f'Model saved in {model_path}')
    return xgb, perf