# -*- coding: utf-8 -*-
"""
Created on Sun Nov 17 22:57:47 2019
https://habr.com/ru/company/ods/blog/325654/
@author: F
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set(style='white')
#%matplotlib inline
from sklearn import decomposition
from sklearn import datasets
from mpl_toolkits.mplot3d import Axes3D

#возьмем набор данных по рукописным цифрам
digits = datasets.load_digits()
X = digits.data
y = digits.target

#%%Вспомним, как выглядят эти цифры – посмотрим на первые десять.
#Картинки здесь представляются матрицей 8 x 8 (интенсивности белого цвета для каждого пикселя). 
#Далее эта матрица "разворачивается" в вектор длины 64, получается признаковое описание объекта.

# f, axes = plt.subplots(5, 2, sharey=True, figsize=(16,6))
plt.figure(figsize=(16, 6))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(X[i,:].reshape([8,8]));
    
#%%Получается, размерность признакового пространства здесь – 64. 
#Но давайте снизим размерность всего до 2 и увидим, что даже на глаз рукописные 
#цифры неплохо разделяются на кластеры.

pca = decomposition.PCA(n_components=2)
X_reduced = pca.fit_transform(X)

print('Projecting %d-dimensional data to 2D' % X.shape[1])

plt.figure(figsize=(12,10))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, 
            edgecolor='none', alpha=0.7, s=40,
            cmap=plt.cm.get_cmap('nipy_spectral', 10))
plt.colorbar()
plt.title('MNIST. PCA projection')

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
plt.title('MNIST. t-SNE projection')

#%%На практике, как правило, выбирают столько главных компонент, 
#чтобы оставить 90% дисперсии исходных данных. 
#В данном случае для этого достаточно выделить 21 главную компоненту, 
#то есть снизить размерность с 64 признаков до 21.
pca = decomposition.PCA().fit(X)

plt.figure(figsize=(10,7))
plt.plot(np.cumsum(pca.explained_variance_ratio_), color='k', lw=2)
plt.xlabel('Number of components')
plt.ylabel('Total explained variance')
plt.xlim(0, 63)
plt.yticks(np.arange(0, 1.1, 0.1))
plt.axvline(21, c='b')
plt.axhline(0.9, c='r')
plt.show();