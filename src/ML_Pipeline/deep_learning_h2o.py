
import numpy as np
import pandas as pd
import h2o

from h2o.estimators.deeplearning import H2ODeepLearningEstimator

def deep_learning_h2o(train_data, predictors_col_names, response_col_name, test_data, model_save_path='output', hidden=[16, 16], epochs=100 ):
    dnn = H2ODeepLearningEstimator(hidden=hidden, epochs=epochs, balance_classes=True)
    dnn.train(x=predictors_col_names, y=response_col_name, training_frame=train_data)
    perf = dnn.model_performance(test_data)
    model_path = h2o.save_model(
            dnn,
            path = model_save_path,
            force = True,
            export_cross_validation_predictions = False)
    print(f'Model saved in {model_path}')
    return dnn , perf