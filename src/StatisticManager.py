import pandas as pd

class StatisticManager:
    def __init__(self):
        pass
    def evaluate(self,pred:pd.Series,ans:pd.Series):
        self.pred=pred
        self.ans=ans
        assert self.pred.size==self.ans.size
        self.size=self.pred.size
        self.acc=0
        for i in range(self.size):
            # print(self.pred[i],self.ans[i])
            if self.pred[i]==self.ans[i]:
                self.acc+=1
            else:
                self.acc/=self.size
        print(self.acc)