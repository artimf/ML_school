# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 15:50:46 2019

@author: 16690918

Q_MATRIX
"""


#%%

import pandas as pd
import numpy as np
# Коэффициент дисконтирования будущих наград
gamma = 0.95
# Параметр скорости обучения
learning_rate = 0.5
# Критерий сходимости алгоритма
tol = 1e-3
experience_buffer = pd.read_csv('Grid_world.csv', dtype = int)
print(experience_buffer[:10])

#%%
#Определение множеств возможных состояний и допустимых действий
# Множество возможных состояний
state_space = sorted(list(set(experience_buffer['s'])))
print('Множество возможных состояний',state_space)

# Множество допустимых действий
action_space = sorted(list(set(experience_buffer['a'])))
print('Множество допустимых действий',action_space)


# Множество допустимых наград
action_space = sorted(list(set(experience_buffer['r'])))
print('Множество допустимых наград',action_space)

#####################################################################
#Создание Q-матрицы
#################### Начало оцениваемого задания ####################
#####################################################################

# Создайте Q-матрицу, в которой строки соответствуют возможным состояниям, а столбцы - допустимым действиям
#Q_matrix = pd.DataFrame(0.0, index=None, columns=None) 
import numpy as np
nda1 = np.zeros(400).reshape(100,4)  
#nda1 = np.arange(400).reshape(100,4) 
Q_matrix = pd.DataFrame(nda1,index=np.arange(1, 101, 1),columns=[1,2,3,4])  
#####################################################################
#################### Конец оцениваемого задания #####################
#####################################################################
# Копия Q-матрицы на предыдущей итерации для проверки сходимости алгоритма
Q_old = Q_matrix.copy()  

#%%
 

continue_condition = True
i = 0
print(Q_matrix[10:23])
# Итерационный цикл
while (continue_condition):

    # Счетчик количества итераций
    i += 1

    # Последовательно обрабатываем каждое наблюдение в experience buffer
    for index, experience in experience_buffer.iterrows():

        # Текущее состояние
        s = experience["s"]

        # Выбранное действие
        a = experience["a"]

        # Полученная награда
        r = experience["r"]

        # Следующее состояние
        s_next = experience["s'"]

        #####################################################################
        #################### Начало оцениваемого задания ####################
        #####################################################################

        # Вычислите значение Q-матрицы при совершении действия a в состоянии s. Используйте метод loc.
        #Q_s_a = None
        Q_s_a = Q_matrix.loc[s,a] 
        #print('>',s,a,Q_s_a,'>',s_next) 

        # Вычислите максимальное значение Q-матрицы в состоянии s_next. Используйте методы loc и max.
        #Q_next_max = None
        Q_next_max = max(Q_matrix.loc[s_next])
        #print('>>>>',s_next,'>',Q_next_max) 

        # Запишите уравнение обновления Q-матрицы. Не забудьте учесть параметр скорости обучения(learning_rate)
        #Q_matrix.loc[s,a] = None 
        Q_matrix.loc[s,a]=(r+Q_s_a+Q_next_max*gamma)*learning_rate
        #print(Q_matrix[10:23])
        #break

        #####################################################################
        #################### Конец оцениваемого задания #####################
        #####################################################################
    #break    
    # Вычисление коэффициента сходимости. После окончания первой итерации он должен быть равен 16009.8365. 
    convergence_rate = (Q_matrix - Q_old).abs().sum().sum()
    print ('Итерация %d завершена. Коэффициент сходимости: %.4f ' % (i, convergence_rate))
    # Проверка выполнения условия сходимости
    if (convergence_rate < tol):
        continue_condition = False
    else:
        Q_old = Q_matrix.copy()

# Завершение алгоритма. Алгоритм должен завершиться после 8 итераций.
print ('\nНеобходимый коэффициент сходимости был достигнут.')
print(Q_matrix)
#%%
# Для каких состояний вывести оптимальные действия?
output_states = [41,42,43,44,45,46,47,48,49,50]
#output_states = np.arange(1, 101, 1)

# Вывод оптимальных действий
optimal_strategy = Q_matrix.idxmax(axis=1).loc[output_states].reset_index()
#print(tabulate(optimal_strategy.values, headers = ['Состояние', 'Оптимальное действие']))
print(optimal_strategy.values)
#%%