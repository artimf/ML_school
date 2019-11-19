# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:57:47 2019
https://habr.com/ru/company/ods/blog/325654/
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
print(X[1,:])

#%%
X_reduced = X#pca.fit_transform(X)
print(X_reduced[:, 1])

plt.figure(figsize=(12,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 6))
plt.colorbar()
plt.title('iris. PCA projection')

#%%Получается, размерность признакового пространства здесь – 11. 
#Но давайте снизим размерность всего до 2 и увидим, что даже на глаз рукописные 
#цифры неплохо разделяются на кластеры.

pca = decomposition.PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print('Projecting %d-dimensional data to 2D' % X.shape[1])

plt.figure(figsize=(12,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 6))
plt.colorbar()
plt.title('iris. PCA projection')
#%%Ну, правда, с t-SNE картинка получается еще лучше, 
#поскольку у PCA ограничение – он находит только линейные комбинации исходных признаков. 
#Зато даже на этом относительно небольшом наборе данных можно заметить, насколько t-SNE дольше работает.
from sklearn.manifold import TSNE
tsne = TSNE(random_state=17)

X_tsne = tsne.fit_transform(X)

plt.figure(figsize=(12,10))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('iris. t-SNE projection')
#%%На практике, как правило, выбирают столько главных компонент, 
#чтобы оставить 90% дисперсии исходных данных. 
#В данном случае для этого достаточно выделить _ главную компоненту, 
#то есть снизить размерность с _ признаков до _.
pca = decomposition.PCA().fit(X)

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 11)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(21, c='b')
plt.axhline(0.9, c='r')
plt.show();
#%%