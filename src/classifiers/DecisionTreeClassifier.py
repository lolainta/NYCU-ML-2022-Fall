import numpy as np
import pandas as pd
from classifiers.BaseClassifier import BaseClassifier
from classifiers.DecisionNode import DecisionNode

target_feature='class_cat'
class DecisionTreeClassifier(BaseClassifier):
    def __init__(self,**kwargs):
        self.root=DecisionNode(None,0)
        super().__init__(kwargs);
    def fit(self,X_train,y_train):
        super().fit(X_train,y_train)
        self.features=self.X_train.columns
        assert X_train.shape[0]==y_train.shape[0],f"Got different size: {X_train.shape[0]},{y_train.shape[0]}"
        self.data=pd.concat([X_train,y_train[target_feature]],axis=1)
        self.root.feed(self.data)
        self.root.genTree(self.features)
    
    def test(self,data:tuple[int,pd.Series]):
        return self.root.test(data[1])
        
    def predict(self, X_test:pd.DataFrame)->pd.Series:
        ret=list()
        for data in X_test.iterrows():
            ret.append(self.test(data))
        return pd.Series(ret)