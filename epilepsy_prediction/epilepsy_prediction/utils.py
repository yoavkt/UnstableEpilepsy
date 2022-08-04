import pandas as pd
import xgboost as xgb


def impute_columns(X, imputed_values):
    """
    Add X with missing columns, the values entered are the mean values

    Args:
        X (pd.DataFrame): The feature matrix
        imputed_values (pd.DataFrame): A the mean feature values of each drug data set

    Returns:
        pd.DataFrame: An imputed data frame
    """
    impute_cols = set(imputed_values.index) - set(X.columns)
    for col in impute_cols:
        X[col] = imputed_values[col]
    return X.loc[:, imputed_values.index]


def load_model(drug_name):
    """
    loads the xgboost model per drug name

    Args:
        drug_name (str): The drug of the xgb model

    Returns:
        xgb.XGBClassifier: An xgb model for unstable epilepsy per the drug
    """
    model = xgb.XGBClassifier()
    clf = f'../models/{drug_name}.xgb'
    model.load_model(clf)
    return model


def data_preprocess(X, impute_drug=None, impute_file_name=f'../csv/mean_pred_cals.csv'):
    """
    load the data and impute it if required

    Args:
        data_file_name (str): The location of a pickle containing  a dictionary with two keys
        one X the data set and another y the outcomes
        impute_drug (str, optional): The drug to impute by Defaults to None.
        impute_file_name (str, optional): the location of the impute data. Defaults to f'../csv/mean_pred_cals.csv'.

    Returns:
        tuple: a tuple where the first element is the imputed data frame and the second is the outcome
    """
    impute_data = pd.read_csv(impute_file_name, index_col=0)
    if not impute_drug is None:
        X = impute_columns(X, impute_data.loc[impute_drug.lower(), :])
    return X.loc[:, impute_data.columns]
