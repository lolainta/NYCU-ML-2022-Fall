import pandas as pd
from classifiers.DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
class StatisticManager:
    def __init__(self):
        pass
    def evaluate(self,pred:pd.Series,true:pd.Series):
        self.pred=pred
        self.true=true['class_cat']
        assert self.pred.size==self.true.size,f"{self.pred.shape},{self.true.shape}"
        self.size=self.pred.size
        self.acc=0
        for yt,yp in zip(self.true,self.pred):
            if yt==yp:
                self.acc+=1
        self.acc*=100/self.size
        print(f"Accuracy: {self.acc}%")

    def crossValidate(self,clf:DecisionTreeClassifier,X:pd.DataFrame,y:pd.Series,cv:int=10):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
        ret=list()
        for i in range(cv):
            clf.fit(X_train,y_train)
            y_pred=clf.predict(X_test)
            self.evaluate(y_pred,y_test)
