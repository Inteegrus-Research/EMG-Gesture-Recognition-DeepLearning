import json
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class SignalScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.medians = None
        self.iqrs = None

    def fit(self, X, y=None):
        X_reshaped = X.reshape(-1, X.shape[-1])
        self.medians = np.median(X_reshaped, axis=0)
        q75, q25 = np.percentile(X_reshaped, [75,25], axis=0)
        self.iqrs = q75 - q25
        self.iqrs[self.iqrs==0]=1.0
        return self

    def transform(self, X):
        return (X - self.medians)/self.iqrs

    def save(self, path):
        json.dump({"medians":self.medians.tolist(),"iqrs":self.iqrs.tolist()}, open(path,"w"))

    def load(self, path):
        data=json.load(open(path,"r"))
        self.medians=np.array(data["medians"])
        self.iqrs=np.array(data["iqrs"])

