import pandas as pd
import numpy as np
import h2o
from h2o.estimators.gbm import H2OGradientBoostingEstimator
from h2o.grid.grid_search import H2OGridSearch


def gb_model_h2o(train_data, predictors_col_names, response_col_name, test_data, model_save_path='output', col_sample_rate=1.0, learn_rate=0.1, max_depth=9, ntrees=500):
    gbm = H2OGradientBoostingEstimator(col_sample_rate=col_sample_rate, learn_rate=learn_rate, max_depth=max_depth, ntrees=ntrees, seed=1234)
    gbm.train(x=predictors_col_names, y=response_col_name, training_frame=train_data)
    perf = gbm.model_performance(test_data)
    model_path = h2o.save_model(
            gbm,
            path = model_save_path,
            force = True,
            export_cross_validation_predictions = False)
    print(f'Model saved in {model_path}')
    return gbm, perf


def gb_model_grid_search(gbm_params, train_data, predictors_col_names, response_col_name, test_data):
    gbm_grid1 = H2OGridSearch(model=H2OGradientBoostingEstimator,
                          grid_id='gbm_grid124',
                          hyper_params=gbm_params)
    gbm_grid1.train(x=predictors_col_names, y=response_col_name, training_frame=train_data)
    gbm_gridperf1 = gbm_grid1.get_grid(sort_by='aucpr', decreasing=True)
    best_gbm1 = gbm_gridperf1.models[0]
    perf = best_gbm1.model_performance(test_data)
    return best_gbm1, perf
    