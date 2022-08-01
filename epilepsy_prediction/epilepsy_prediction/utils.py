import pandas as pd
from os import path
import xgboost as xgb

def get_imputation_data():
    df 
    return df

def impute_columns(X,imputed_values):
    impute_cols = set(imputed_values.index) - set(X.columns)
    for col in impute_cols:
        X[col] = imputed_values[col]
    return X.loc[:,imputed_values.index]

def load_model(drug_name):
    model = xgb.XGBClassifier()
    clf = f'../models/{drug_name}.xgb'
    model.load_model(clf)
    return model

def load_data(data_file_name,impute_drug=None,impute_file_name = f'../csv/mean_pred_cals.csv'):
    test_df=pd.read_pickle(data_file_name)
    impute_data = pd.read_csv(impute_file_name,index_col=0)
    if impute_drug is None:
        X = test_df['X'].loc[:,impute_data.columns]
    else:
        X = impute_columns(test_df['X'],impute_data.loc[impute_drug.lower(),:])
    return X,test_df['y']