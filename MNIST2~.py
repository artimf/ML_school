"""
Created on Thu Oct 31 17:33:30 2019
стр 99
@author: F
"""

from sklearn.datasets import load_digits
import pylab as pl

digits = load_digits()
print('print(len(digits.images))',len(digits.images))

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
"""
images_and_predictions = list(zip(digits.images[:3],fit.predict(X)))
for index, (image,prediction) in enumerate(images_and_predictions[:8]):
    plt.subplot(6,3,index+5)
    plt.axis('off')
    plt.imshow(image,cmap=plt.cm.gray_r,interpolation='nearest')
    plt.title('Prediction: %i' % prediction) 
plt.show()
"""