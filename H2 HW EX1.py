# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 14:21:42 2020

@author: Zangiacomi Sandro
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m



########QUESTION 1#############################################################



Food_consumption = pd.read_csv ('food-consumption.csv')
mean_data=Food_consumption.mean(axis=0)
m_points=len(Food_consumption.index)


Row=[]
for row in range(0,len(Food_consumption.index)):
    Row.append(np.array(Food_consumption.iloc[row,1:].tolist())-np.asanyarray(mean_data).T)
    
X=np.asarray(Row)
Xt=X.T

Cov=(1/m_points)*Xt.dot(X)


# # ############### food=features######################
k=2
def Eigen_Value_Vect_selected(data_set,k):
    Value, Vector=np.linalg.eig(data_set)
    Vector=np.real(Vector)
    Value=np.real(Value)  
    value_sorted=sorted(Value.tolist(),reverse=1) 
    Eigen_Vectors_Selected=[]
    for eigVal in value_sorted[0:k]:
        Eigen_Vectors_Selected.append(Vector[:,Value.tolist().index(eigVal)])
    return np.asarray(Eigen_Vectors_Selected), value_sorted[0:k]

Eigen_vector, Eigen_values=Eigen_Value_Vect_selected(Cov,k)

reduced_representation=[]

for data_point in Row:      
    reduced_representation.append([Eigen_vector[i].dot(data_point)/m.sqrt(Eigen_values[i]) for i in range(0,k)])
plt.subplot(2, 1, 1) 
plt.title('Eigen Vectror 1')   
plt.stem(list(Eigen_vector)[0],linefmt=':') 
plt.subplot(2, 1, 2)
plt.title('Eigen Vectror 2') 
plt.stem(list(Eigen_vector)[1]) 

plt.figure()
plt.title('Eigen Vectors for Countries=Features')
plt.show        

countries=Food_consumption.iloc[:,0].tolist()
 
##########PLOT QUESTION 4 #################################################################
X=[]
Y=[]
for points in reduced_representation:
    X.append(points[0])
    Y.append(points[1])
   
count=0 
plt.figure()
plt.scatter(X,Y,s=10) 
for points in reduced_representation:
    plt.annotate(countries[count], xy = (points[0],points[1]))
    count+=1
plt.title(' Countries representation using their two principal components')                                      
plt.show() 
   
##############countries=features QUESTION 4##############################################

Food_consumption2=Food_consumption.transpose()
Food_consumption2=Food_consumption2.drop(['Country'], axis=0)
Food_consumption2=Food_consumption2.astype(float)
Food_consumption2.columns=Food_consumption.iloc[:,0].tolist()
m_points2=len(Food_consumption2.index)

mean_data2=Food_consumption2.mean(axis=0)
Row2=[]
for row in range(0,len(Food_consumption2.index)):
    Row2.append(np.array(Food_consumption2.iloc[row,:].tolist())-np.asanyarray(mean_data2).T)
    
X2=np.asarray(Row2)
Xt2=X2.T

Cov2=(1/m_points2)*Xt2.dot(X2)



Eigen_vector2, Eigen_values2=Eigen_Value_Vect_selected(Cov2,k)
reduced_representation2=[]
for data_point in Row2:      
    reduced_representation2.append([np.vdot(Eigen_vector2[i],data_point)/m.sqrt(Eigen_values2[i]) for i in range(0,k)])
        

Food=Food_consumption2.index.tolist()

X2=[]
Y2=[]
for points in reduced_representation2:   
    X2.append(points[0])
    Y2.append(points[1])
plt.figure()    
count=0 
plt.scatter(X2,Y2,s=10) 

for points in reduced_representation2:
    plt.annotate(Food[count], xy = (points[0],points[1]))
    count+=1
plt.title(' Food representation using their two principal components')    
plt.show() 

plt.figure()
plt.subplot(2, 1, 1) 
plt.title('Eigen Vectror 1')   
plt.stem(list(Eigen_vector2)[0],linefmt=':') 
plt.subplot(2, 1, 2)
plt.title('Eigen Vectror 2')
plt.stem(list(Eigen_vector2)[1]) 

#plt.savefig('Eigen Vectors for Food=Features.png')
plt.show()
