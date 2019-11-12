# -*- coding: utf-8 -*-
"""
Created on Thu Oct 31 17:33:30 2019
стр 99
@author: F
"""
#%%
from sklearn.datasets import load_digits
import pylab as pl
#%%

digits = load_digits()
print('print(len(digits.images))',len(digits.images))

pl.gray()
pl.matshow(digits.images[1]) 
pl.show()

#print(digits.images[1].reshape(1,-1))
#%%
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pylab as plt
#%%
y=digits.target
print('print(len(y))', len(y))
print('print(y[:20])', (y[:20]))
#%%
n_samples = len(digits.images)
X= digits.images.reshape((n_samples,-1)) 
print("print(n_samples)",n_samples)
print('print(len(X))',len(X))  
#%%
X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
gnb = GaussianNB()
fit=gnb.fit(X_train,y_train)
predicted=fit.predict(X_test)
#print('###',fit.predict(digits.images[44].reshape(1,-1)))
print('print(y_test)',y_test)
print('print(predicted)',predicted)
#%%
confusion_matrix(y_test,predicted)
print(confusion_matrix(y_test,predicted))
print('сумма всех предсказаний ',confusion_matrix(y_test,predicted).sum())#сумма всех предсказаний
print('количесвто верных предсказаний ',confusion_matrix(y_test,predicted).trace())#количесвто верных предсказаний
#%%
images_and_predictions = list(zip(digits.images,fit.predict(X)))
for index, (image,prediction) in enumerate(images_and_predictions[:8]):
    plt.subplot(6,3,index+5)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Prediction: %i' % prediction) 
plt.show()
#%%
import pickle
filename = 'MNIST_finalized_model.sav'
pickle.dump(fit, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))
predicted=loaded_model.predict(digits.images[1].reshape(1,-1))
print('#####',predicted) 
#%%
"""
"""
