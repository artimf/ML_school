# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:33:30 2019
стр 110
Кластер
@author: F
"""
#%%
from warnings import simplefilter# import warnings filter
simplefilter(action='ignore', category=FutureWarning)# ignore all future warnings
import sklearn
from sklearn import cluster
import pandas as pd

data = sklearn.datasets.load_iris()
X=pd.DataFrame(data.data,columns=list(data.feature_names))
print(X[:5])
#%%
model=cluster.KMeans(n_clusters=3,random_state=25)
results=model.fit(X)
X["cluster"]=results.predict(X)
X["target"]=data.target
X["c"]="lookatmeIamImportant"
print(X[:5])
#%%
classification_result=X[["cluster","target","c"]].groupby(["cluster","target"]).agg("count")
print(classification_result)
#%%