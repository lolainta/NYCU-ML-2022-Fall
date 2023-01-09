import pandas as pd
import numpy as np
import math

target_feature='class_cat'

class DecisionNode:
    def __init__(self,parent,id:int):
        self.id=id
        self.parent=parent
        # self.leftChild=None
        # self.rightChild=None
        self.leaf=False
        # print(f"Node contructed id={self.id}")
    def feed(self,data:pd.DataFrame):
        self.data=data
        self.size=self.data.shape[0]
        self.orgEntropy=self.entropy(self.data)
        if self.orgEntropy==0:
            print(f"{self.id}: Leaf node")
            self.leaf=True
            # print(self.data)
            self.label=self.data[target_feature][self.data[target_feature].index[0]]
            # print("start lable")
            # print(self.label)
            # print("end lable")
    def genTree(self,features:list[str]):
        if self.leaf==True:
            return
        self.features=features
        self.conditions=self.conditions(features)
        chose=np.random.randint(len(self.conditions),size=1)[0]
        self.condition=self.conditions[chose]
        self.leftChild=DecisionNode(self,2*self.id+1)
        self.rightChild=DecisionNode(self,2*self.id+2)
        ldata,rdata=self.split(self.condition[0],self.condition[1])
        print(f"{self.id}: {self.condition},{self.maxig},{ldata.shape},{rdata.shape}")
        # print(ldata,rdata,sep='\n')
        self.leftChild.feed(ldata)
        self.rightChild.feed(rdata)
        self.leftChild.genTree(self.features)
        self.rightChild.genTree(self.features)
        # self.leftChild.feed(ldata[features],ldata[target_featrue])
        # self.rightChild.feed(rdata[features],rdata[target_featrue])
    
    def conditions(self,features:list[str])->list[tuple[str,float]]:
        self.maxig=float("-inf")
        argmax=list()
        for feat in features:
            threshold=self.threasholds(feat)
            # print(f"Data found {len(thresholds)} thresholds with feature: {feat}")
            for thres in threshold:
                curig=self.ig(feat,thres)
                if curig>self.maxig:
                    argmax.clear()
                    self.maxig=curig
                    argmax.append((feat,thres))
                elif curig==self.maxig:
                    argmax.append((feat,thres))
        # print(argmax,maxig)
        return argmax
    
    def threasholds(self,feature:str)->list():
        ret=list()
        self.data.sort_values(by=[feature],ignore_index=True,inplace=True)
        for i in range(self.data.shape[0]-1):
            assert self.data[feature][i]<=self.data[feature][i+1],f"Internal Error: feature={feature},i={i},{self.data[feature][i]}>{self.data[feature][i+1]}"
            if self.data[target_feature][i]!=self.data[target_feature][i+1] and self.data[feature][i]!=self.data[feature][i+1]:
                ret.append((self.data[feature][i]+self.data[feature][i+1])/2)
        # print(self.data[feature])
        return ret
    
    def ig(self,feature:str,threashold:float)->float:
        ret=self.orgEntropy
        ldata,rdata=self.split(feature,threashold)
        tmp1=ldata[target_feature].count()
        tmp2=rdata[target_feature].count()
        assert tmp1+tmp2==self.size
        ret-=tmp1/self.size*self.entropy(ldata)
        ret-=tmp2/self.size*self.entropy(rdata)
        return ret

    def entropy(self,data:pd.DataFrame)->float:
        ret=float(0)
        for ratio in data[target_feature].value_counts(normalize=True):
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