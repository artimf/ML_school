"""
Created on Thu Oct 31 17:33:30 2019
стр 99
@author: F
"""

#from sklearn.datasets import load_digits
from sklearn.utils import Bunch
import pylab as pl
import numpy as np
from os.path import dirname, exists, expanduser, isdir, join, splitext

def loaddigits(n_class=10, return_X_y=False):
    module_path = dirname(__file__)
    data = np.loadtxt(join(module_path, 'my', 'my2.csv'),
                          delimiter=',')
    with open(join(module_path, 'my', 'digits.rst')) as f:
        descr = f.read()
    target = data[:, -1].astype(np.int, copy=False)
    print('>>>',len(target))
    #0,0,5,13,9,1,0,0,0,0,13,15,10,15,5,0,0,3,15,2,0,11,8,0,0,4,12,0,0,8,8,0,0,5,8,0,0,9,8,0,0,4,11,0,1,12,7,0,0,2,14,5,10,12,0,0,0,0,6,13,10,0,0,0,0

    flat_data = data[:, :-1]
    images = flat_data.view()
    images.shape = (-1, 2, 2)

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


digits = loaddigits()
n=33
print('>>>@',digits.target[n])
print(digits.images[n])
pl.gray()
pl.matshow(digits.images[n]) 
pl.show()

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pylab as plt

y=digits.target
print('print(len(y))', len(y))
print('print(y[:20])', (y[60:80]))

n_samples = len(digits.images)
X= digits.images.reshape((n_samples,-1)) 
print("print(n_samples)",n_samples)
print('print(len(X))',len(X))  

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
gnb = GaussianNB()
fit=gnb.fit(X_train,y_train)
predicted=fit.predict(X_test) 
confusion_matrix(y_test,predicted)
print(confusion_matrix(y_test,predicted))
print('сумма всех предсказаний ',confusion_matrix(y_test,predicted).sum())#сумма всех предсказаний
print('количесвто верных предсказаний ',confusion_matrix(y_test,predicted).trace())#количесвто верных предсказаний

import pickle
filename = 'finalized_model.sav'
pickle.dump(fit, open(filename, 'wb'))
loaded_model = pickle.load(open(filename, 'rb'))

pp=np.array([2,2,2,2]).reshape(1,-1)
pp=np.array([3,3,3,3]).reshape(1,-1) 
print(pp)
predicted=loaded_model.predict(pp) 
print('#####',predicted)
images_and_predictions = list(zip(digits.images,fit.predict(X)))
for index,(image,prediction) in enumerate(images_and_predictions[2:6]):
    print(image.reshape(1,-1),';', str(prediction))
#plt.show()

#%%
import tabulate as tb
#%%
"""
pl.gray()
pl.matshow(digits.images[1]) 
pl.show()
print('>>>',digits.images[1]) 
print(digits.images[1].reshape(1,-1))
x=digits.images[1]

from PIL import Image
import numpy as np

img = Image.open( "2.jpg" )
img = img.resize((8,8))
img = img.convert('L')
img.load()
dataa = np.asarray( img, dtype="float64" )
print(dataa)
print(dataa.reshape(1,-1))
xxx=dataa.reshape(1,-1)
pl.matshow(dataa) 
pl.show()

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
import pylab as plt

y=digits.target
print('print(len(y))', len(y))
print('print(y[:20])', (y[:20]))

n_samples = len(digits.images)
X=  digits.images.reshape((n_samples,-1)) 
print("print(n_samples)",n_samples)
print('print(len(X))',len(X))  

X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=0)
gnb = GaussianNB()
fit=gnb.fit(X_train,y_train)
predicted=fit.predict(X_test)
  


print('@@@',X_test[1])
print(digits.images[1].reshape(1,-1))
#print('>>>',fit.predict(xxx))
print('###',fit.predict(digits.images[44].reshape(1,-1)))
confusion_matrix(y_test,predicted)
pl.gray()
pl.matshow(digits.images[44]) 
pl.show()
#pp=fit.predict(xxx)
#cm=confusion_matrix(y_test,pp)

#print(cm)
#print(confusion_matrix(y_test,predicted).sum())#сумма всех предсказаний
#print(confusion_matrix(y_test,predicted).trace())#количесвто верных предсказаний

#images_and_predictions = list(zip(digits.images,fit.predict(X)))

images_and_predictions = list(zip(digits.images[:3],fit.predict(X)))
for index, (image,prediction) in enumerate(images_and_predictions[:8]):
    plt.subplot(6,3,index+5)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Prediction: %i' % prediction) 
plt.show()
"""