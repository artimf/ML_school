# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:33:30 2019
стр 118
Тренеровка Прецептрона на наблюдениях
@author: F
"""
#%%
import numpy as np
class preceptron():
    def __init__(self,X,y, threshold = 0.5, learning_rate =0.1, max_epochs=10):
        self.threshold=threshold
        self.learning_rate = learning_rate
        self.X = X
        self.y = y
        self.max_epochs = max_epochs

    def initialize(self, init_type = 'zeros'):
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
                #print(epoch,X,y,error_count)
            if error_count== 0:
                print("Traing successful")
                break
            if epoch >= self.max_epochs:
                print("reached maximum epochs, no perfect prediction")
                break

    def train_observation(self,X,y,error_count):
        result = np.dot(X,self.weights)>self.threshold
        error= y-result
        print('error',y,'-',result,'=',y-result,'|',X,'*',self.weights,'=',np.dot(X,self.weights))
        if error !=0:
            error_count += 1
            for index, value in enumerate(X):
                xx=(self.weights[index],'+|',self.learning_rate ,error ,value,'=',self.learning_rate *error *value )
                self.weights[index] += self.learning_rate *error *value
                print('###',X,y,';',xx,'>>',self.weights)
        return error_count

    def predict(self,X): 
        print('>>>>>',X,self.weights,np.dot(X,self.weights))
        return int(np.dot(X,self.weights)>self.threshold)

X=[(1,0,0),(1,1,0),(1,1,1),(1,1,1),(1,0,1),(1,0,1)]
y=[1,1,0,0,1,1]

X=[(1,0,0),(1,1,0)]
y=[1,1]

p=preceptron(X,y)
p.initialize() 
p.train()
print(p.predict((1,0,0)))
print(p.predict((1,1,0))) 
print(p.predict((0,1,1)))
print(p.predict((0,0,1))) 
#%%
p.X=[(0,1,1),(0,0,1)]
p.y=[1,1]
 
p.train()
print(p.predict((1,0,0)))
print(p.predict((1,1,0))) 
print(p.predict((0,1,1)))
print(p.predict((0,0,1)))