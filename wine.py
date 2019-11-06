# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:33:30 2019
стр 99
@author: F
"""


print('_----------- load_wine')
from sklearn.datasets import load_wine
wine = load_wine()
print(wine.target[[10, 70, 140]]) 
print(wine.target_names)
print(wine.data[1])
print(wine.feature_names)

n_samples = len(wine.data)
X= wine.data.reshape((n_samples,-1)) 
y=wine.target
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
gnb = GaussianNB()
fit=gnb.fit(X_train,y_train)
predicted=fit.predict(X_test) 
confusion_matrix(y_test,predicted)
print(confusion_matrix(y_test,predicted))
print('сумма всех предсказаний ',confusion_matrix(y_test,predicted).sum())#сумма всех предсказаний
print('количесвто верных предсказаний ',confusion_matrix(y_test,predicted).trace())#количесвто верных предсказаний

pp=np.array([13.27,4.28,2.26,20,120,1.59,0.69,0.43,1.35,10.2,0.59,1.56,835]).reshape(1,-1)
print('>>>>',fit.predict(pp))

#for key,value in wine.items():
#    print(key,'\n',value,'\n')
print('data.shape\t',wine['data'].shape,
      '\ntarget.shape \t',wine['target'].shape)

import pandas as pd
features = pd.DataFrame(data=wine['data'],columns=wine['feature_names'])
data = features
data['target']=wine['target']
data['class']=data['target'].map(lambda ind: wine['target_names'][ind])
print(data.head())
print(data.describe())
import seaborn as sns
sns.distplot(data['alcohol'],kde=0)

import matplotlib.pyplot as pltx
for i in data.target.unique():
    sns.distplot(data['alcohol'][data.target==i],
                 kde=1,label='{}'.format(i))
pltx.legend()

import matplotlib.gridspec as gridspec
for feature in wine['feature_names']:
    if feature=='proline':
        print(feature)
        #sns.boxplot(data=data,x=data.target,y=data[feature])
        gs1 = gridspec.GridSpec(3,1)
        ax1 = plt.subplot(gs1[:-1])
        ax2 = plt.subplot(gs1[-1])
        gs1.update(right=0.60)
        sns.boxplot(x=feature,y='class',data=data,ax=ax2)
        sns.kdeplot(data[feature][data.target==0],ax=ax1,label='0')
        sns.kdeplot(data[feature][data.target==1],ax=ax1,label='1')
        sns.kdeplot(data[feature][data.target==2],ax=ax1,label='2')
        ax2.yaxis.label.set_visible(False)
        ax1.xaxis.set_visible(False)
        pltx.show()
#images_and_predictions = list(zip(wine.data,fit.predict(X)))
#for index, (image,prediction) in enumerate(images_and_predictions[2:6]):
#    print(image.reshape(1,-1),';', str(prediction))
