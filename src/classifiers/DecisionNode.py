import pandas as pd
import numpy as np
import math

class DecisionNode:
    def __init__(self,parent,id:int,target_feature:str):
        self.id=id
        self.parent=parent
        self.leaf=False
        self.target_feature=target_feature
        # print(f"Node contructed id={self.id}")

    def feed(self,data:pd.DataFrame):
        self.data=data
        self.size=self.data.shape[0]
        self.orgEntropy=self.entropy(self.data)
        if self.orgEntropy==0:
            # print(f"{self.id}: Leaf node")
            self.leaf=True
            # self.label=self.data[self.target_feature][self.data[self.target_feature].index[0]]
            self.label=self.data[self.target_feature].value_counts().idxmax()

    def genTree(self,features:list[str],depth:int):
        # print(f"id:{self.id}, depth:{depth}")
        if self.leaf==True:
            return
        if depth==0:
            # print(f"Max depth id={self.id}")
            self.leaf=True
            self.label=self.data[self.target_feature].value_counts().idxmax()
            # print(self.label)
            return
        self.features=features
        self.conds=self.conditions(features)
        chose=np.random.randint(len(self.conds),size=1)[0]
        self.condition=self.conds[chose]
        if self.condition[1]<=0.1 and False:
            print(f"Low IG id={self.id}")
            self.leaf=True
            self.label=self.data[self.target_feature].value_counts().idxmax()
            # print(self.label)
            return
        self.leftChild=DecisionNode(self,2*self.id+1,self.target_feature)
        self.rightChild=DecisionNode(self,2*self.id+2,self.target_feature)
        ldata,rdata=self.split(self.condition[0],self.condition[1])
        # print(f"{self.id}: {self.condition},{self.maxig},{ldata.shape},{rdata.shape}")
        self.leftChild.feed(ldata)
        self.rightChild.feed(rdata)
        self.leftChild.genTree(self.features,depth-1)
        self.rightChild.genTree(self.features,depth-1)
    
    def conditions(self,features:list[str])->list[tuple[str,float]]:
        self.maxig=float("-inf")
        argmax=list()
        for feat in features:
            threshold=self.threasholds(feat)
            for thres in threshold:
                curig=self.ig(feat,thres)
                if curig>self.maxig:
                    argmax.clear()
                    self.maxig=curig
                    argmax.append((feat,thres))
                elif curig==self.maxig:
                    argmax.append((feat,thres))
        return argmax
    
    def threasholds(self,feature:str)->list():
        ret=list()
        self.data.sort_values(by=[feature],ignore_index=True,inplace=True)
        for i in range(self.data.shape[0]-1):
            assert self.data[feature][i]<=self.data[feature][i+1],f"Internal Error: feature={feature},i={i},{self.data[feature][i]}>{self.data[feature][i+1]}"
            if self.data[self.target_feature][i]!=self.data[self.target_feature][i+1] and self.data[feature][i]!=self.data[feature][i+1]:
                ret.append((self.data[feature][i]+self.data[feature][i+1])/2)
        # print(self.data[feature])
        return ret
    
    def ig(self,feature:str,threashold:float)->float:
        ret=self.orgEntropy
        ldata,rdata=self.split(feature,threashold)
        tmp1=ldata[self.target_feature].count()
        tmp2=rdata[self.target_feature].count()
        assert tmp1+tmp2==self.size
        ret-=tmp1/self.size*self.entropy(ldata)
        ret-=tmp2/self.size*self.entropy(rdata)
        return ret

    def entropy(self,data:pd.DataFrame)->float:
        ret=float(0)
        for ratio in data[self.target_feature].value_counts(normalize=True):
            if ratio:
                ret+=ratio*math.log2(ratio)
        return -ret
    
    def split(self,feature:str,threashold:float):
        ldata=self.data[self.data[feature]<threashold]
        rdata=self.data[self.data[feature]>threashold]
        return ldata,rdata

    def test(self,data:pd.Series):
        if self.leaf==True:
            return self.label
        # print(f"Search in id={self.id}, condition={self.condition}")
        if data[self.condition[0]]<self.condition[1]:
            return self.leftChild.test(data)
        else:
            return self.rightChild.test(data)