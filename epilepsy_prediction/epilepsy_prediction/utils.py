import pandas as pd
import xgboost as xgb
import pickle
from collections import OrderedDict
from fuse.eval.metrics.metrics_common import CI
from fuse.eval.metrics.classification.metrics_classification_common import (
    MetricAUCROC,
)
from fuse.eval.metrics.classification.metrics_model_comparison_common import (
    MetricDelongsTest,
    MetricMcnemarsTest,
)
from fuse.eval.evaluator import EvaluatorDefault

from fuse.eval.metrics.metrics_common import  CI

seed = 23423465
metrics = OrderedDict([
                ("auc1", CI(MetricAUCROC(pred="pred1", target="target"), 
                       stratum="target", rnd_seed=seed)),
            ("auc2", CI(MetricAUCROC(pred="pred2", target="target"), 
                       stratum="target", rnd_seed=seed)),
            ("mcnemar_test", 
            MetricMcnemarsTest(pred1="pred1", pred2="pred2",target="target")),
            ('delong_test',
            MetricDelongsTest(pred1="pred_proba1", pred2="pred_proba2",target="target"))
        ])

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


def load_prediction_model(drug_name):
    """
    loads the xgboost model per drug name

    Args:
        drug_name (str): The drug of the xgb model

    Returns:
        xgb.XGBClassifier: An xgb model for unstable epilepsy per the drug
    """
    model = xgb.XGBClassifier()
    clf = f'../models/prediction/{drug_name}.xgb'
    model.load_model(clf)
    return model

def load_imputation_model(drug_name):
    """_summary_

    Args:
        drug_name (_type_): _description_
    """
    file_name = f'../models/prediction/imputer_{drug_name}.pkl'
    file = open(file_name,'rb')
    imputer = pickle.load(file)
    file.close()
    return imputer

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

def prepare_fuse_data(clf,clf2,X,y):
    res = pd.DataFrame(columns=['pred','target','id'])
    res['id']=X.index
    res['pred1'] = clf.predict(X).squeeze()
    res['pred2'] = clf2.predict(X).squeeze()
    res['pred_proba1'] = clf.predict_proba(X)[:,1].squeeze()
    res['pred_proba2'] = clf2.predict_proba(X)[:,1].squeeze()
    res['target'] = y.astype(int).values  
    return res

def fuse_string(results):
    sum = []
    for k,v in results['metrics'].items():
        if 'auc' in k:
            sum.append(k + f": {v['mean']:.2f} (upper: {v['conf_upper']:.2f}, lower: {v['conf_lower']:.2f}) \n ")
        if 'mcnemar' in k or 'delong' in k:
            sum.append(k + f": p-value {v['p_value']:.3f} \n ")
    return ''.join(sum)

def evaluate_model(clf1,clf2,X,y):
    evaluator = EvaluatorDefault()
    return evaluator.eval(ids=None, data=prepare_fuse_data(clf1,clf2,X,y), metrics=metrics) 