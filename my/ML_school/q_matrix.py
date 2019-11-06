"""
В этом задание мы спрогнозируем норму сбережения M2-M0 байесовским методом, используя вероятностный язык программирования Stan. В задании нужно будет смоделировать сезонную компоненту.

Stan - высокопроизводительный фреймворк для байесовских моделей и не только. На нем удобно сформулировать модель в виде уравнений, и далее программа методом точечной оптимизации, MCMC, или Variational Inference оценивает параметры с высокой производительностью. От конкурентов его отличает очень хорошая реализация алгоритма HMC (Hamilton Markov Chain). Подробнее о Stan - https://mc-stan.org/users/documentation/.

Stan manual - https://github.com/stan-dev/stan/releases/download/v2.17.1/stan-reference-2.17.1.pdf
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("1")
data = pd.read_csv(r"data_saving_rate.csv",sep=';',index_col='Date', decimal=',')
data.index = pd.to_datetime(data.index, format='%d.%m.%Y')
data = data.dropna(axis=1)#удалить null ось 1
print(data.head())

data['M_growth'] = data['M2-M0'][1:] - data.shift(1)['M2-M0'][1:] # прирост M2-M0 будет нашей целевой переменной
data = data.dropna(axis=0)
print(data.head())

#Создадим файл с кодом на языке stan. Допишите кусок кода. '//' - комментарии в Stan
#скомпилируем код stan в машинный код
#python -m pip install pystan --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.org
#https://pystan.readthedocs.io/en/latest/windows.html
#conda config --set ssl_verify false 
#conda install libpython m2w64-toolchain -c msys2
#conda install pystan 
import pystan
model = pystan.StanModel(file='model.stan')

#разобьем данные на тестовую и обучающую выборку
factor = data.M_growth
y = data.Savings_rate
 
length = len(y)
factor_study = factor.head(int(2 / 3 * length))
factor_test = factor.tail(length-int(2 / 3 * length))

y_study = y.head(int(2 / 3 * length))
y_test = y.tail(length-int(2 / 3 * length))


#подготовим данные для stan
N = len(y_study)
P = len(y_test)
D = 1
y_stan = np.array(y_study)
x_stan = [np.array(factor)]
data_stan = dict(N=N, D=D, P=P, y=y_stan, x=x_stan)

#обучим модель и сделаем прогноз
#Мы применяем метод Markov Chain Monte Carlo для симуляций из постериорного распределения параметров.
#В данном случае мы делаем 2000 симуляций и 2 попытки.
#В модели могут всплыть предупреждения о сходимости и т.п. Так как в алгоритме MCMC много параметров управления,
#которые следует вдумчиво подобирать. Здесь мы просто знакомимся с методом. 
#fitModel = model.sampling(data=data_stan, iter=200, chains=2)
fitModel = model.sampling(data=data_stan, iter=1, chains=1)

#Построим графики
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.plot(y, c='green')  # факт
yfittedAndForecasted = np.mean( np.array(fitModel.extract()['predY']), axis = 0) #fitted_value + прогноз
yforecasted = yfittedAndForecasted[-len(y_test):]# прогноз
plt.plot(y_test.index, yforecasted, c='red')
plt.show()

stanModRmse = np.sqrt(np.mean((yforecasted - y[-len(y_test):])**2))
stanModRmse

naiveModelRmse = np.sqrt(np.mean(( np.mean(y_test) - y[-len(y_test):] )**2)) # модель, в которой прогноз целевой переменной равен среднему значению предыдущих.
naiveModelRmse

#Проверка!
stanModRmse < naiveModelRmse
 
