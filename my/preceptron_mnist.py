# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:33:30 2019
стр 118
Тренеровка Прецептрона на наблюдениях
@author: F
"""
#%%Локальная функция загрузки данных
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import confusion_matrix
from os.path import dirname, exists, expanduser, isdir, join, splitext
from sklearn.utils import Bunch
import pylab as pl
import pylab as plt
def loaddigits2(n_class=10, return_X_y=False):
    module_path = dirname(__file__)
    data = np.loadtxt(join(module_path, 'data', 'digits.csv.gz'),
                          delimiter=',')
    with open(join(module_path, 'data', 'digits.rst')) as f:
        descr = f.read()
    target = data[:, -1].astype(np.int, copy=False)
    print('target>>>',len(target))
    #0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0,0

    flat_data = data[:, :-1]
    images = flat_data.view()
    images.shape = (-1, 8, 8)
    print('image>>>', len(images))

    if n_class < 10:
        idx = target < n_class
        flat_data, target = flat_data[idx], target[idx]
        images = images[idx]

    if return_X_y:
        return flat_data, target

    return Bunch(data=flat_data,
                 target=target,
                 target_names=np.arange(10),
                 images=images,
                 DESCR=descr)

#%%
import numpy as np
class preceptron():
    def __init__(self,X,y, threshold = 7000, learning_rate =0.1, max_epochs=200):
        self.threshold=threshold
        self.learning_rate = learning_rate
        self.X = X
        self.y = y
        self.max_epochs = max_epochs

    def initialize(self, init_type = 'random'):
        if init_type == 'random':
            self.weights = np.random.rand(len(self.X[0])) * 0.05
        if init_type == 'zeros':
            self.weights = np.zeros(len(self.X[0]))

    def train(self):
        epoch=0
        while True:
            error_count = 0
            epoch += 1
            for (X,y) in zip(self.X, self.y):
                error_count += self.train_observation(X,y,error_count)
                #print(epoch,X.reshape(8,8),y,error_count)
                """
                print('weights>>>',self.weights.reshape(8,8),sum(self.weights))
                print('X>>>',X.reshape(8,8),sum(X)) 
                pl.matshow(X.reshape(8,8)) 
                pl.show()
                print('___')
                pl.matshow(self.weights.reshape(8,8)) 
                pl.show()
                """
            if error_count== 0:
                print("Traing successful",epoch)
                break
            if epoch >= self.max_epochs:
                print("reached maximum epochs, no perfect prediction")
                break

    def train_observation(self,X,y,error_count):
        result = np.dot(X,self.weights)>self.threshold
        #print(">>>>>>>",np.dot(X,self.weights),self.threshold,result)
        error= y-result
        #print('error',y,'-',result,'=',y-result,'|',X,'*',self.weights,'=',np.dot(X,self.weights))
        if error !=0:
            error_count += 1
            for index, value in enumerate(X):
                xx=(self.weights[index],'+|',self.learning_rate ,error ,value,'=',self.learning_rate *error *value)
                self.weights[index] += self.learning_rate *error *value
                #print('###',X,y,';',xx,'>>',self.weights)
        return error_count

    def predict(self,X): 
        #print('>>>>>',X,self.weights,np.dot(X,self.weights))
        #print('>>>',np.dot(X,x),self.threshold)
        return int(np.dot(X,self.weights)>self.threshold)

d2 = loaddigits2()
o=np.where(d2.target==1)
oo=sorted(list(set(o[0])))
print(len(oo))
X=d2.images[1].reshape(1,-1)
y= [1] * 1 
p=preceptron(X,y)
p.initialize() 
p.train()
#print('Predict',p.predict(d2.images[4].reshape(1,-1)))
print('p.weights>>>',sum(p.weights))
#%%
d2 = loaddigits2()
X= d2.images[1].reshape(1,-1)
print(d2.images[1])
y= [1] * 1
p=preceptron(X,y)
p.initialize() 
X_train=oo
for i in X_train:
    p.X=d2.images[i].reshape(1,-1)
    p.y=[1] * d2.target[i]
    p.train()
    #print(i,X,y)
print('p.weights>>>',sum(p.weights))
#%%
n=170
pl.gray()
pl.matshow(d2.images[X_train[n]]) 
pl.show()
print('p.weights>>>',sum(p.weights))
print(d2.target[X_train[n]],p.predict(d2.images[X_train[n]].reshape(1,-1))) 
#%%
n=0
pl.gray()
pl.matshow(d2.images[n]) 
pl.show()
print('p.weights>>>',sum(p.weights))
print(d2.target[n],p.predict(d2.images[n].reshape(1,-1))) 

#%%
for n in range(90): 
    print(d2.target[n],p.predict(d2.images[n].reshape(1,-1))) 
