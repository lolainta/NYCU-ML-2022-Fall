import numpy as np
import pandas as pd

class BaseClassifier:
    def __init__(self,target_feature,random_state=72510):
        self.seed=random_state
        self.target_feature=target_feature['target_feature']
    def fit(self,X_train,y_train):
        self.X_train=X_train
        self.y_train=y_train
        
    def predict(self,X_test):
        pass