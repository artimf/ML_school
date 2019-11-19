"""
Created on Thu Oct 31 17:33:30 2019
стр 99
@author: F
Контролируемое обучение
#from sklearn.datasets import load_wine
#wine = load_wine()
"""
#%%
from warnings import simplefilter# import warnings filter
simplefilter(action='ignore', category=FutureWarning)# ignore all future warnings
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

data = load_breast_cancer() 
"""
Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=feature_names,
                 filename=csv_filename)
"""
n_samples = len(data.data)
y=data.target
X=data.data.reshape((n_samples,-1))
print(list(data.target_names) )
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
max_scores = dict([max(scores.items(), key=lambda k_v: k_v[1])])
print('best',max_scores)

#%%  

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
import seaborn as sns
import matplotlib.pyplot as pltx
data = load_breast_cancer() 
print(data['feature_names'])

features = pd.DataFrame(data=data['data'],columns=data['feature_names']) 
datax = features
datax['target']=data['target']
datax['class']=datax['target'].map(lambda ind: data['target_names'][ind])
print(datax.head())
#print(datax.describe())
#%%
import matplotlib.pyplot as pltx
for i in datax.target.unique():
    sns.distplot(datax['worst compactness'][datax.target==i],kde=1,label='{}'.format(i))
pltx.legend()
pltx.show()
#%%
import matplotlib.gridspec as gridspec
for feature in data['feature_names']:
    if feature=='worst compactness':
        print(feature)
        #sns.boxplot(datax=data,x=datax.target,y=data[feature])
        gs1 = gridspec.GridSpec(3,1)
        ax1 = plt.subplot(gs1[:-1])
        ax2 = plt.subplot(gs1[-1])
        gs1.update(right=0.60)
        sns.boxplot(x=feature,y='class',data=datax,ax=ax2)
        sns.kdeplot(datax[feature][datax.target==0],ax=ax1,label='0')
        sns.kdeplot(datax[feature][datax.target==1],ax=ax1,label='1')
        sns.kdeplot(datax[feature][datax.target==2],ax=ax1,label='2')
        ax2.yaxis.label.set_visible(False)
        ax1.xaxis.set_visible(False)
        
        pltx.show()

#%%