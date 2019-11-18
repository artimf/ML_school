# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:33:30 2019
стр 102
Не Контролируемое обучение
метод главных компонент
@author: F
"""

#%%
import pandas as pd
from sklearn import preprocessing
from  sklearn.decomposition import PCA
import pylab as plt
#url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv'
url ='./data/winequality-red.csv'

data=pd.read_csv(url,sep=";")
X= data[[u'fixed acidity',u'volatile acidity',u'citric acid',u'residual sugar',u'chlorides',u'free sulfur dioxide',u'total sulfur dioxide',u'density',u'pH',u'sulphates',u'alcohol']]
XX=X
y=data.quality
X= preprocessing.StandardScaler().fit(X).transform(X)
#%%

model= PCA()
results = model.fit(X)
Z=results.transform(X)
plt.plot(results.explained_variance_)
plt.show()

#%%
hidenVar=pd.DataFrame(results.components_, columns=list(
        [u'fixed acidity', u'volatile acidity', u'citric acid', u'residual sugar', 
         u'chlorides', u'free sulfur dioxide', u'total sulfur dioxide', u'density',
         u'pH', u'sulphates', u'alcohol']))
print(hidenVar)
"""

print(list(hidenVar))
print('@',(hidenVar.iloc[0:1]).sum())
print((hidenVar.iloc[0:1]).sum().sum())

nn=XX[:1]
print('nn>',nn.sum())
print('XX>',(nn.sum()*(hidenVar.iloc[0])))
print('XX>>>',(nn*(hidenVar.iloc[0])).sum().sum())

print((hidenVar.iloc[0][0]))
"""

#%%
ind,sum=0,0

for p in XX[0:1].sum():
    print(p,hidenVar.iloc[0][ind],p*hidenVar.iloc[0][ind])
    sum=sum+p*hidenVar.iloc[0][ind]
    ind=ind+1
print(sum)
#%%
#прогнозирование качества вина до применнеия анализа главных компонент
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
gnb = GaussianNB()
fit = gnb.fit(X,y)
pred=fit.predict(X)
print(confusion_matrix(pred,y)) 
print('сумма всех предсказаний ',confusion_matrix(pred,y).sum())#сумма всех предсказаний
print('количесвто верных предсказаний ',confusion_matrix(pred,y).trace())#количесвто верных предсказаний
print(accuracy_score(pred, y))
#%%
#прогнозирование качества вина с наращиванием количества гланых компонент
predicted_correct =[]
for i in range(1,10):
    model=PCA(n_components=i)
    results=model.fit(X)
    Z=results.transform(X)
    fit=gnb.fit(Z,y)
    pred=fit.predict(Z)
    predicted_correct.append(confusion_matrix(pred,y).trace())
    print(predicted_correct,accuracy_score(pred, y))
plt.plot(predicted_correct)
plt.show()
#%%