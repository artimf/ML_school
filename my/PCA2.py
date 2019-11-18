# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:33:30 2019
стр 102
Не Контролируемое обучение
метод главных компонент
@author: F
"""
from sklearn.utils import Bunch
import pylab as pl
from os.path import dirname, exists, expanduser, isdir, join, splitext
import csv
 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white') 
from sklearn import decomposition
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

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

def load_w(return_X_y=False): 
    module_path = 'C:/Anaconda3/envs/test/my'#dirname(__file__)
    print(module_path)
    data, target, target_names = load_data(module_path, 'winequality-red_test.csv')
    csv_filename = join(module_path, 'data', 'winequality-red_test.csv')

    with open(join(module_path, 'data', 'wine_data.rst')) as rst_file:
        fdescr = rst_file.read()

    if return_X_y:
        return data, target

    return Bunch(data=data, target=target,
                 target_names=target_names,
                 DESCR=fdescr,
                 feature_names=['fixed acidity','volatile acidity','citric acid','residual sugar','chlorides','free sulfur dioxide','total sulfur dioxide','density','pH','sulphates','alcohol'],
                 filename= csv_filename)
 
#%% 
# Загрузим
iris = load_w()
X = iris.data
y = iris.target
print(y[:24])
#%%

# Заведём красивую трёхмерную картинку
fig = plt.figure(1, figsize=(6, 5))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()

for name, label in [('3_',3), ('4_', 4), ('5_', 5), ('6_',6),('7_',7),('8_',8)]: 
    ax.text3D(X[y == label, 3].mean(),
              X[y == label, 4].mean()+ 1.5,
              X[y == label, 5].mean(),
              #X[y == label, 6].mean(),
              #X[y == label, 7].mean(),
              #X[y == label, 8].mean(),
              name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
print(y)


#Поменяем порядок цветов меток, чтобы они соответствовали правильному
y_clr = np.choose(y, y[:9]).astype(np.float)

ax.scatter(X[:, 3], X[:, 4], X[:,5], X[:,6], X[:,7], c=y_clr, cmap=plt.cm.nipy_spectral)

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

#%%
#Теперь посмотрим, насколько PCA улучшит результаты для модели, которая 
#в данном случае плохо справится с классификацией из-за того, 
#что у неё не хватит сложности для описания данных

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
# Выделим из наших данных валидационную выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3,  stratify=y )

# Для примера возьмём неглубокое дерево решений
#clf = DecisionTreeClassifier(max_depth=2, random_state=42)
#clf.fit(X_train, y_train)
#preds = clf.predict_proba(X_test)
clf = GaussianNB()
fit = clf.fit(X_train,y_train)
preds=fit.predict(X_test)
#print('Accuracy: {:.5f}'.format(accuracy_score(y_test, preds.argmax(axis=1))))
#print(accuracy_score(preds, y_test))
print(accuracy_score(y_test, preds))
#%%
#прогнозирование качества вина с наращиванием количества гланых компонент
from  sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
predicted_correct =[]
for i in range(1,10):
    model=PCA(n_components=i)
    results=model.fit(X)
    Z=results.transform(X)
    fit=clf.fit(Z,y)
    pred=fit.predict(Z)
    predicted_correct.append(confusion_matrix(pred,y).trace())
   # print(predicted_correct,accuracy_score(pred, y))
plt.plot(predicted_correct)
plt.show()
#%%Теперь попробуем сделать то же самое, но с данными, для которых мы снизили размерность до 2D
# 
 

#Без pca
X_pca =(X) 
# И нарисуем получившиеся точки в нашем новом пространстве
plt.plot(X_pca[y == 3, 0], X_pca[y == 3, 1], 'bo', label='3')
plt.plot(X_pca[y == 4, 0], X_pca[y == 4, 1], 'go', label='4')
plt.plot(X_pca[y == 5, 0], X_pca[y == 5, 1], 'ro', label='5')
plt.plot(X_pca[y == 6, 0], X_pca[y == 6, 1], 'yo', label='6')
plt.plot(X_pca[y == 7, 0], X_pca[y == 7, 1], 'bo', label='7')
plt.plot(X_pca[y == 8, 0], X_pca[y == 8, 1], 'ro', label='8')
plt.legend(loc=0);
plt.show()
####Прогоним встроенный в sklearn PCA
pca = decomposition.PCA(n_components=7)
X_centered = X - X.mean(axis=0)
pca.fit(X_centered) 
X_pca = pca.transform(X_centered) 
# И нарисуем получившиеся точки в нашем новом пространстве
plt.plot(X_pca[y == 3, 0], X_pca[y == 3, 1], 'bo', label='3')
plt.plot(X_pca[y == 4, 0], X_pca[y == 4, 1], 'go', label='4')
plt.plot(X_pca[y == 5, 0], X_pca[y == 5, 1], 'ro', label='5')
plt.plot(X_pca[y == 6, 0], X_pca[y == 6, 1], 'yo', label='6')
plt.plot(X_pca[y == 7, 0], X_pca[y == 7, 1], 'bo', label='7')
plt.plot(X_pca[y == 8, 0], X_pca[y == 8, 1], 'ro', label='8')
plt.legend(loc=0);
 
# Повторим то же самое разбиение на валидацию и тренировочную выборку.
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=.3,  stratify=y )

#clf = DecisionTreeClassifier(max_depth=2, random_state=42)
#clf.fit(X_train, y_train)
#preds = clf.predict_proba(X_test)
clf = GaussianNB()
fit = clf.fit(X_train,y_train)
preds=fit.predict(X_test)
#print('Accuracy: {:.5f}'.format(accuracy_score(y_test, preds.argmax(axis=1))))
print(accuracy_score(y_test, preds))
#%%Посмотрим на 2 главные компоненты в последнем PCA-представлении данных 
#и на тот процент исходной дисперсии в даных, который они "объясняют".

for i, component in enumerate(pca.components_):
    print("{} component: {}% of initial variance".format(i + 1, 
          round(100 * pca.explained_variance_ratio_[i], 2)))
    print(" + ".join("%.3f x %s" % (value, name)
                     for value, name in zip(component, iris.feature_names)))

#%%
 