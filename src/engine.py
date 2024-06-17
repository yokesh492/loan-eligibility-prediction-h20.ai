import numpy as np
import pandas as pd
import h2o
from ML_Pipeline.utils import read_dataset_advanced, replace_outliers_with_nan, club_duplicate_rows, map_entries, replace_values, replace_values_with_nan
from ML_Pipeline.impute import impute
from ML_Pipeline.categorical_encoding import categorical_encoding
from ML_Pipeline.knn_imputation import knn_imputation
from ML_Pipeline.data_scaling import data_scaling
from ML_Pipeline.train_test_h2o import train_test_h2o
from ML_Pipeline.gb_model_h2o import gb_model_h2o, gb_model_grid_search
from ML_Pipeline.xgb_model_h2o import xgb_model_h2o
from ML_Pipeline.deep_learning_h2o import deep_learning_h2o
from ML_Pipeline.light_gbm_h2o import light_gbm_h2o
h2o.init()

data = read_dataset_advanced('../input/LoansTrainingSetV2.csv', sort_by_col="Loan ID")
loan_amount_threshold = data['Current Loan Amount'].max() / 2.0
data = replace_outliers_with_nan(data, 'Current Loan Amount', threshold_val=loan_amount_threshold)
data = club_duplicate_rows(data, id_col="Loan ID")
assert data['Loan ID'].unique().shape[0] == data.shape[0]

data = pd.read_csv('../input/grouped_by_loan_id.csv')
# outliers elimination in credit score column 
data["Credit Score"] = np.where(data["Credit Score"] > 800, data["Credit Score"] / 10, data["Credit Score"])
data = impute(data, 'Credit Score', method='value', value=data["Credit Score"].describe()['75%'])

years_dict = {'< 1 year':0, '1 year':1, '2 years':2, '3 years':3, '4 years':4, '5 years':5, 
              '6 years':6, '7 years':7, '8 years':8, '9 years':9, '10+ years':10}

data = map_entries(data, col='Years in current job', mapping_dict=years_dict)
data = impute(data, 'Years in current job', method='value', value=data['Years in current job'].describe()['75%'])
data = impute(data, 'Months since last delinquent', method='value', value=data['Months since last delinquent'].describe()['75%'])
term_dict = {'Short Term':0, 'Long Term':1}
data = map_entries(data, col="Term", mapping_dict=term_dict)
# 'HaveMortgage' and 'Home Mortgage' are the same. So replace 'HaveMortgage' with  'Home Mortgage'
data = replace_values(data, col='Home Ownership', original_value='HaveMortgage', replace_with='Home Mortgage')
data = replace_values(data, col='Purpose', original_value='Other', replace_with='other')
data = replace_values(data, col='Monthly Debt', original_value='$', replace_with='')
data['Monthly Debt'] = pd.to_numeric(data['Monthly Debt'] )
data = replace_values_with_nan(data, col='Maximum Open Credit', original_value='#VALUE!')
data = impute(data, col='Maximum Open Credit', method='median')
data['Maximum Open Credit'] = pd.to_numeric(data['Maximum Open Credit'])
data.loc[data['Maximum Open Credit'] > 171423, 'Maximum Open Credit'] = 171423
data = impute(data, col='Monthly Debt', method='median')
data = impute(data, col='Bankruptcies', method='median')
data = impute(data, 'Tax Liens', method='value', value=0.0)
data = categorical_encoding(data, 'Purpose', 'purpose')
data = categorical_encoding(data, col='Home Ownership', prefix='home')
column_names_to_impute = ['Current Loan Amount', 'Annual Income']
data = knn_imputation(data, column_names_to_impute=column_names_to_impute)
data['fin_propensity'] = data['Annual Income'] / (12 * data['Monthly Debt'] + 1)
# print(data.isnull().sum())

final_df = pd.read_csv('../input/final_data.csv')
column_names_to_standardize = list(final_df.columns[3:])
final_df = data_scaling(data=final_df, column_names_to_standardize=column_names_to_standardize)
# print(final_df.head())
train, test = train_test_h2o(final_df)

predictors = list(final_df.columns[3:])
response = "Loan Status"

# Gradient Boost Model

gbm, perf = gb_model_h2o(train_data=train, predictors_col_names=predictors, response_col_name=response, test_data=test)
print(perf)

# Gradient boost grid search

gbm_params1 = {'learn_rate': [0.01, 0.1],
                'max_depth': [5, 9],
                'ntrees': [300, 500],
                'sample_rate': [0.8, 1.0],
                'col_sample_rate': [0.5, 1.0]}
            
gbm, perf = gb_model_grid_search(gbm_params=gbm_params1, train_data=train, predictors_col_names=predictors, 
                                response_col_name=response, test_data=test)


# XGB Model

xgb, perf = xgb_model_h2o(train_data=train, predictors_col_names=predictors, response_col_name=response, test_data=test)
print(perf)

# Deep Learning Model

dnn , perf = deep_learning_h2o(train_data=train, predictors_col_names=predictors, response_col_name=response, test_data=test)
print(perf)

# LightGBM Model
lgbm, perf = light_gbm_h2o(train_data=train, predictors_col_names=predictors, response_col_name=response, test_data=test)
print(perf)

