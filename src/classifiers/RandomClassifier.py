import numpy as np
import pandas as pd
from classifiers.BaseClassifier import BaseClassifier

class RandomClassifier(BaseClassifier):
    def fit(self,X_train,y_train):
        super().fit(X_train,y_train)
        
    def predict(self,X_test):
        self.y_pred=np.random.random(X_test.shape[0])
        return pd.DataFrame(self.y_pred)