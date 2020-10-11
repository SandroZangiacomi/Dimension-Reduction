# -*- coding: utf-8 -*-
"""
Created on Fri Sep  4 21:34:03 2020

@author: Zangiacomi Sandro
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m
import scipy.io
from PIL import Image


mat = scipy.io.loadmat('isomap.mat')
data_set=pd.DataFrame(mat['images'])
############### Question a) ###############

def euclidian_distance(x,y): #Takes data frame
    x=np.asarray(x)
    y=np.asarray(y)
    return m.sqrt(np.vdot((x-y).T,x-y))

def Creation_Adjency(data_set,Dist):
    Matrix_size =len(data_set.iloc[0,:])
    Adjacency_Matrix=np.zeros(shape=(Matrix_size,Matrix_size))
    for i in range(0,Matrix_size):
        for j in range(0,Matrix_size):
            vector1=data_set.iloc[:,i]
            vector2=data_set.iloc[:,j]
            distance=Dist(vector1,vector2)
            Adjacency_Matrix[i][j]=distance
    return Adjacency_Matrix

def Adjency(Creation_Adjency,treshold):
    
    for i in range(0,len( Creation_Adjency[:][0])):
        for j in range(0,len( Creation_Adjency[:][0])):
            if Creation_Adjency[i][j]>treshold:
                Creation_Adjency[i][j]= 1000000
           
    return Creation_Adjency



############### END Question a) ###############


##############QUESTION b)######################
import networkx as nx

def Matrix_D(W):
    # Generate Graph and Obtain Matrix D, \\
    # from weight matrix W defining the weight on the edge between each pair of nodes.
    # Note that you can assign sufficiently large weights to non-existing edges.
    n = np.shape(W)[0]
    Graph = nx.DiGraph()
    for i in range(n):
        for j in range(n):
            Graph.add_weighted_edges_from([(i,j,min(W[i,j], W[j,i]))])

    res = dict(nx.all_pairs_dijkstra_path_length(Graph))
    D = np.zeros([n,n])
    for i in range(n):
        for j in range(n):
            D[i,j] = res[i][j]
    np.savetxt('D.csv', D)
    return D


def C_matrix(D):
    H=np.eye(len(D)) - np.ones((len(D),len(D)))*1/len(D)
    D=D*D
    a=-0.5*H.dot(D)
    b=a.dot(H)
    return b  

def Eigen_Value_Vect_selected(data_set,k):
    Value, Vector=np.linalg.eig(data_set)
    Vector=np.real(Vector)
    Value=np.real(Value)  
    value_sorted=sorted(Value.tolist(),reverse=1)
    Eigen_Vectors_Selected=[]
    for eigVal in value_sorted[0:k]:
        Eigen_Vectors_Selected.append(Vector[:,Value.tolist().index(eigVal)])
    return np.asarray(Eigen_Vectors_Selected), value_sorted[0:k]



def Reduced_Z(Eigen_vector, Eigen_values):
    Z=np.zeros(shape=(len(Eigen_values),len(Eigen_values)))
    Z[0][0]=Eigen_values[0]**(0.5)
    Z[1][1]=Eigen_values[1]**(0.5)
    
    
    return (Eigen_vector.T).dot(Z)
    


images=[]  
nbr_im=len(data_set.iloc[0,:])
for i in range(0,nbr_im):
    im=np.asarray([data_set.iloc[:,i].tolist()])
    images.append(np.reshape(im,(64,64)))
    
    

def threshold(Creation_Adjency):
    B=Creation_Adjency.copy()
    Max=[]
    for i in range(0,len(B[:][0])):
        a=sorted(B[:,i].tolist())[101]
        Max.append(a)
    return max(Max)
        
    
    
Adjency_dist=Creation_Adjency(data_set,euclidian_distance)
e=threshold(Adjency_dist)
AdjencyMarix=Adjency(Adjency_dist,e)

A_to_show=AdjencyMarix.copy()
for i in range(0,len(A_to_show)):
    for j in range(0,len(A_to_show)):
        if A_to_show[i][j]>e+1:
            A_to_show[i][j]=1
        else :
            A_to_show[i][j]=A_to_show[i][j]/(int(e)+1)
            
plt.figure()            
plt.imshow(A_to_show)
plt.colorbar()
plt.show()

D=Matrix_D(AdjencyMarix)                 
C_matrix1= C_matrix(D)
Eigen_vector, Eigen_values=Eigen_Value_Vect_selected(C_matrix1,2)
   
Z=  Reduced_Z(Eigen_vector, Eigen_values)  
Z=Z.tolist()    
    
X=[]
Y=[]
points_selected=[0,50,100,150,200,300,350,400,450,500,600]

for points in Z:
    X.append(points[0])
    Y.append(points[1])
plt.figure()
for i in points_selected:
    plt.annotate(str(i),(X[i],Y[i]))
plt.title('ISOMAP with Euclidian Distance')
plt.scatter(X,Y,s=8)
plt.show()
    



img1 = Image.fromarray(np.uint8(images[0]* 255).T , 'L')
img2 = Image.fromarray(np.uint8(images[50]* 255).T , 'L')
img3 = Image.fromarray(np.uint8(images[100]* 255).T , 'L')
img4 = Image.fromarray(np.uint8(images[150]* 255).T , 'L')
img5 = Image.fromarray(np.uint8(images[200]* 255).T , 'L')
img6 = Image.fromarray(np.uint8(images[300]* 255).T , 'L')
img7 = Image.fromarray(np.uint8(images[350]* 255).T , 'L')
img8 = Image.fromarray(np.uint8(images[400]* 255).T , 'L')
img9 = Image.fromarray(np.uint8(images[450]* 255).T , 'L')
img10 = Image.fromarray(np.uint8(images[500]* 255).T , 'L')
img11 = Image.fromarray(np.uint8(images[600]* 255).T , 'L')


  

  

############QUESTION c)#######################
def manhattan_dist(x,y):
    x=np.asarray(x)
    y=np.asarray(y)
    a=abs(x-y)
    a=a.tolist()
    return sum(a)
    

Adjency_l1=Creation_Adjency(data_set,manhattan_dist)
e1=threshold(Adjency_l1)
AdjencyMarix_l1=Adjency(Adjency_l1,e1)

D2=Matrix_D(AdjencyMarix_l1)                 
C_matrix2=C_matrix(D2)
Eigen_vector2, Eigen_values2=Eigen_Value_Vect_selected(C_matrix2,2)

Z2=Reduced_Z(Eigen_vector2, Eigen_values2)  
Z2=Z2.tolist()    
    
X2=[]
Y2=[]
for points in Z:
    X2.append(points[0])
    Y2.append(points[1])  
plt.figure()   
for i in points_selected:
    plt.annotate(str(i),(X2[i],Y2[i]))    
a=plt.scatter(X2,Y2,s=10)
plt.title('ISOMAP performed with l1 distance')
plt.show()

####Question d)####################

mean_data=data_set.mean(axis=1)
Row=[]
for row in range(0,len(data_set.iloc[0,:])):
    Row.append(np.array(data_set.iloc[:,row].tolist())-np.asanyarray(mean_data).T)
    
X=np.asarray(Row)
Xt=X.T

Cov=(1/len(data_set.iloc[0,:]))*Xt.dot(X)

Eigen_vector3, Eigen_values3=Eigen_Value_Vect_selected(Cov,2)

reduced_representation=[]

for data_point in Row:      
    reduced_representation.append([Eigen_vector3[i].dot(data_point)/m.sqrt(Eigen_values3[i]) for i in range(0,2)])
    
X3=[]
Y3=[]
for points in reduced_representation:
    X3.append(points[0])
    Y3.append(points[1])
plt.figure()   
for i in points_selected:
    plt.annotate(str(i),(X3[i],Y3[i]))
plt.scatter(X3,Y3,s=10) 
plt.title('PCA Method')
plt.show()
    
    
    
    
    
    
    