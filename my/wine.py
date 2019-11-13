"""
Created on Thu Oct 31 17:33:30 2019
стр 99
@author: F
Контролируемое обучение
#from sklearn.datasets import load_wine
#wine = load_wine()
"""
#%%
from sklearn.utils import Bunch
import pylab as pl
import numpy as np
from os.path import dirname, exists, expanduser, isdir, join, splitext
import csv

def load_data(module_path, data_file_name):
    with open(join(module_path, 'data', data_file_name)) as csv_file:
        data_file = csv.reader(csv_file)
        temp = next(data_file)
        n_samples = int(temp[0])
        n_features = int(temp[1])
        target_names = np.array(temp[2:])
        data = np.empty((n_samples, n_features))
        target = np.empty((n_samples,), dtype=np.int)
        print(temp)
        print(n_samples,n_features)
        print(target_names)  

        for i, ir in enumerate(data_file):
            data[i] = np.asarray(ir[:-1], dtype=np.float64)
            target[i] = np.asarray(ir[-1], dtype=np.int)

    return data, target, target_names

def loadwine(return_X_y=False):
    module_path = dirname(__file__)
    data, target, target_names = load_data(module_path,  'wine_data.csv')

    with open(join(module_path, 'data', 'wine_data.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['alcohol',
                                'malic_acid',
                                'ash',
                                'alcalinity_of_ash',
                                'magnesium',
                                'total_phenols',
                                'flavanoids',
                                'nonflavanoid_phenols',
                                'proanthocyanins',
                                'color_intensity',
                                'hue',
                                'od280/od315_of_diluted_wines',
                                'proline'])

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

wine = loadwine()
n_samples = len(wine.data)
 
y=wine.target
X= wine.data.reshape((n_samples,-1))
#%%
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
#methods
print("===== полученные размерности =====")
print("X_train.shape:", X_train.shape)
print("X_test.shape:", X_test.shape)
print("y_train.shape:", y_train.shape)
print("y_test.shape:", y_test.shape)

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.naive_bayes import GaussianNB
rfc = RandomForestClassifier()
lr = LogisticRegression()
svc = SVC(kernel="linear")
knn = KNN(n_neighbors=1)
nb = GaussianNB()
scores = {}
for name, clf in [("random forest", rfc), 
                  ("logistic regression", lr),
                  ("SVM", svc),
                  ("knn", knn),
                  ("naive bayes", nb)
                 ]:
        if name == "xnaive bayes": 
            b=1
            #clf.fit(X_train.toarray(), y_train)
            #scores[name] = accuracy_score(y_test, clf.predict(X_test.toarray()))
        else:
            clf.fit(X_train, y_train)
            scores[name] = accuracy_score(y_test, clf.predict(X_test))
        print(name, scores[name])
#%% 
for k, v in scores.items(): print(k, v)
max_scores = dict([max(scores.items(), key=lambda k_v: k_v[1])])
print('best',max_scores)

from sklearn.metrics import confusion_matrix
fit=rfc.fit(X_train,y_train)
predicted=fit.predict(X_test)
confusion_matrix(y_test,predicted)
print(confusion_matrix(y_test,predicted))

print('сумма всех предсказаний ',confusion_matrix(y_test,predicted).sum())#сумма всех предсказаний
print('количесвто верных предсказаний ',confusion_matrix(y_test,predicted).trace())#количесвто верных предсказаний

#%%
import pandas as pd
import pylab as plt
features = pd.DataFrame(data=wine['data'],columns=wine['feature_names'])
data = features
data['target']=wine['target']
data['class']=data['target'].map(lambda ind: wine['target_names'][ind])
print(data.head())
print(data.describe())

import seaborn as sns
print(wine['feature_names'])
print(data.target.unique())
import matplotlib.pyplot as pltx
for i in data.target.unique():
    sns.distplot(data['magnesium'][data.target==i],kde=1,label='{}'.format(i))
pltx.legend()
pltx.show()

import matplotlib.gridspec as gridspec
for feature in wine['feature_names']:
    if feature=='magnesium':
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

#%%