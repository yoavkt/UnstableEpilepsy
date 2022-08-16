
from copy import deepcopy
from tabnanny import verbose
from typing_extensions import Self
from sklearn.linear_model import LassoCV
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
class column_imputer:
    def __init__(self,file_name,impute_drug,add_mean_values=True) -> None:
        self.file_name = file_name
        self.impute_drug = impute_drug
        self.impute_data = pd.read_csv(file_name, index_col=0).loc[impute_drug,:]
        self.add_mean_values = add_mean_values
        self.impute_order = None
        self.missing_cols = None
        pass

    def fit(self,X):
        self.missing_cols = set(self.impute_data.index) - set(X.columns)
        self.impute_order = X.columns
        return None

    def transform(self,X):
        for col in self.missing_cols:
            if self.mean_values:
                X[col] = self.impute_data[col]
            else:
                X[col] = np.nan
        return X[self.impute_order]

    def fit_transform(self, X):
        self.fit(X)
        return self.fit_transform(X)
class regress_missing_imputer:
    """

    """
    def __init__(self,numeric_estimator=None,binary_estimator=None,missing_cols=None,ignore_cols=None):
        if numeric_estimator is None:
            numeric_estimator = LassoCV()
        self._base_numeric_estimator = numeric_estimator
        if binary_estimator is None:
            binary_estimator = GridSearchCV(LogisticRegression(penalty='elasticnet',solver='saga'),
                                            param_grid={'l1_ratio':np.arange(0,1,0.1)})
        self._base_binary_estimator = binary_estimator
        self.missing_cols = missing_cols
        self.estimator_dict = {}
        self.ignore_cols=list(ignore_cols)+list(missing_cols)
        return None

    def _get_data_set(self,X,target,isnull=False):
        train_X = X.loc[X[target].isnull()==isnull,~X.columns.isin(self.ignore_cols)]
        train_y = X.loc[X[target].isnull()==isnull,target]
        return train_X,train_y
    
    def _get_estimator(self,X,target):
        if X[target].dtype == 'bool':
            return deepcopy(self._base_binary_estimator)
        else:
            return deepcopy(self._base_numeric_estimator)

    def fit(self,X):
        for col in self.missing_cols:
            est = self._get_estimator(X,col)
            Xt,yt = self._get_data_set(X,col)
            est.fit(Xt,yt)
            self.estimator_dict[col] = est
        return Self

    def transform(self,X):
        for col in self.missing_cols:
            if any(X[col].isnull()):
                X_pred,_ = self._get_data_set(X,col,isnull=True)
                X.loc[X[col].isnull(),col] = self.estimator_dict[col].predict(X_pred)
        return X

    def fit_transform(self, X):
        self.fit(X)
        return self.fit_transform(X)