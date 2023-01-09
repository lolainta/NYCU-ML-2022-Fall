import pandas as pd
from classifiers.DecisionTreeClassifier import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report

class StatisticManager:
    def __init__(self,target_feature):
        self.target_feature=target_feature

    def report(self,pred:pd.Series,true:pd.Series):
        self.pred=pred
        self.true=true[self.target_feature]
        assert self.pred.size==self.true.size,f"{self.pred.shape},{self.true.shape}"
        self.size=self.pred.size
        self.accuracy()
        print(f"Accuracy: {self.acc}%")
        # print(f"Accurary: ")
        self.confusionMatrix=confusion_matrix(self.true,self.pred)
        self.display()
        report=classification_report(self.true,self.pred,zero_division=0)
        print(report)

    def display(self):
        disp=ConfusionMatrixDisplay(self.confusionMatrix)
        disp.plot()
    
    def accuracy(self):
        self.acc=0
        for yt,yp in zip(self.true,self.pred):
            if yt==yp:
                self.acc+=1
        self.acc*=100/self.size
        return self.acc
    def evaluate(self,clf:DecisionTreeClassifier,X:pd.DataFrame,y:pd.Series,portion:float=0.3):
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=portion)
        clf.fit(X_train,y_train)
        y_pred=clf.predict(X_test)
        self.report(y_pred,y_test)
