# -*- coding: utf-8 -*-
"""
Created on Sun Sep  6 17:43:53 2020

@author: Zangiacomi Sandro
"""
from PIL import Image
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy import signal


Subject01_1 = Image.open("yalefaces\\subject01.glasses.gif")
Subject01_2 = Image.open("yalefaces\\subject01.happy.gif")
Subject01_3 = Image.open("yalefaces\\subject01.leftlight.gif")
Subject01_4 = Image.open("yalefaces\\subject01.noglasses.gif")
Subject01_5 = Image.open("yalefaces\\subject01.normal.gif")
Subject01_6 = Image.open("yalefaces\\subject01.rightlight.gif")
Subject01_7 = Image.open("yalefaces\\subject01.sad.gif")
Subject01_8 = Image.open("yalefaces\\subject01.sleepy.gif")
Subject01_9 = Image.open("yalefaces\\subject01.surprised.gif")
Subject01_10 = Image.open("yalefaces\\subject01.wink.gif")

Subject02_1 = Image.open("yalefaces\subject02.glasses.gif")
Subject02_2 = Image.open("yalefaces\\subject02.happy.gif")
Subject02_3 = Image.open("yalefaces\\subject02.leftlight.gif")
Subject02_4 = Image.open("yalefaces\\subject02.noglasses.gif")
Subject02_5 = Image.open("yalefaces\\subject02.normal.gif")
Subject02_6 = Image.open("yalefaces\\subject02.rightlight.gif")
Subject02_7 = Image.open("yalefaces\\subject02.sad.gif")
Subject02_8 = Image.open("yalefaces\\subject02.sleepy.gif")
Subject02_9 = Image.open("yalefaces\\subject02.wink.gif")

Subject1=[Subject01_1,Subject01_2,Subject01_3,Subject01_4,Subject01_5,Subject01_6,
          Subject01_7,Subject01_8,Subject01_9,Subject01_10]
Subject2=[Subject02_1,Subject02_2,Subject02_3,Subject02_4,Subject02_5,Subject02_6,
          Subject02_7,Subject02_8,Subject02_9]

def reducing(subject_list):
    Reduced_image_Subject=[]
    
    count=0
    
    for i in subject_list:
        Subject1[count]=np.asarray(i)
        count+=1
    
    for img in subject_list:
        red_im=signal.decimate(np.asarray(img).astype(float),4,n=None,ftype='iir',axis=-1,zero_phase=True)
        red_im=signal.decimate(red_im.T,4,n=None,ftype='iir',axis=-1,zero_phase=True)
        Reduced_image_Subject.append(list(red_im.T.reshape(80*61)))
    Reduced_image_Subject=np.asarray(Reduced_image_Subject)
    
    return np.asarray(Reduced_image_Subject)

Array_Image_Sub1=reducing(Subject1)
Array_Image_Sub2=reducing(Subject2)
    
Subject1_DF=pd.DataFrame(Array_Image_Sub1)
Subject2_DF=pd.DataFrame(Array_Image_Sub2)

Row_Subject1=[]
Row_Subject2=[]
mean_Subject1=Subject1_DF.mean(axis=0)
mean_Subject2=Subject2_DF.mean(axis=0)

for row in range(0,len(Subject1_DF.index)):
    Row_Subject1.append(np.array(Subject1_DF.iloc[row,:].tolist())-np.asanyarray(mean_Subject1).T)
    
for row in range(0,len(Subject2_DF.index)):
    Row_Subject2.append(np.array(Subject2_DF.iloc[row,:].tolist())-np.asanyarray(mean_Subject2).T)
    
X_Subject1=np.asarray(Row_Subject1)
Xt_Subject1=X_Subject1.T

X_Subject2=np.asarray(Row_Subject2)
Xt_Subject2=X_Subject2.T


Cov_Subject1=(1/len(Subject1_DF.index))*Xt_Subject1.dot(X_Subject1)
Cov_Subject2=(1/len(Subject2_DF.index))*Xt_Subject2.dot(X_Subject2)

def Eigen_Value_Vect_selected(data_set,k):
    Value, Vector=np.linalg.eig(data_set)
    Vector=np.real(Vector)
    Value=np.real(Value)  
    value_sorted=sorted(Value.tolist(),reverse=1) 
    Eigen_Vectors_Selected=[]
    for eigVal in value_sorted[0:k]:
        Eigen_Vectors_Selected.append(Vector[:,Value.tolist().index(eigVal)].tolist())
    return Eigen_Vectors_Selected, value_sorted[0:k]

def eigenfaces(k):
    Eigen_vector_S1, Eigen_values_S1=Eigen_Value_Vect_selected(Cov_Subject1,k)
    Eigen_vector_S2, Eigen_values_S2=Eigen_Value_Vect_selected(Cov_Subject2,k)
    
    Eigenfaces1=[]
    Eigenfaces2=[]
    
    for i in range (0,k):
        Eigenfaces1.append(Eigen_vector_S1[i])
        Eigenfaces2.append(Eigen_vector_S2[i])
        
    return  Eigenfaces1,  Eigenfaces2
    
Eigenfaces1, Eigenfaces2=eigenfaces(6)
for i in Eigenfaces1:
    plt.imshow(np.reshape(np.asarray(i),(61,80)),cmap="gray")
    plt.show()


 ########QUESTION a)##################   
    
fig=plt.figure(figsize=(8, 8))
columns = 2
rows = 3

for i in range(0, columns*rows):
    fig.add_subplot(rows, columns, i+1)
    plt.imshow(np.reshape(np.asarray(Eigenfaces1[i]),(61,80)),cmap='gray')
fig2=plt.figure(figsize=(8, 8))    
for i in range(0, columns*rows):
    fig2.add_subplot(rows, columns, i+1)
    plt.imshow(np.reshape(np.asarray(Eigenfaces2[i]),(61,80)),cmap='gray')

#####################QUESTION b#######################
    



Subject01_test = [Image.open("yalefaces\\subject01-test.gif")]
Subject02_test = [Image.open("yalefaces\\subject02-test.gif")]

Reduced_image_Subject1=reducing(Subject01_test)
Reduced_image_Subject2=reducing(Subject02_test)        

def scoresub(eigenface,test):
    score=np.vdot(test,np.multiply(eigenface,255))/(np.linalg.norm(np.multiply(eigenface,255))*np.linalg.norm(test))
    return abs(score)


s1_1=scoresub(Eigenfaces1[0], Reduced_image_Subject1)   #0.8723046574090426
s1_2=scoresub(Eigenfaces1[0], Reduced_image_Subject2)   #0.6943034367602281
s2_2=scoresub(Eigenfaces2[0], Reduced_image_Subject2)   #0.4059097290512246
s2_1=scoresub(Eigenfaces2[0], Reduced_image_Subject1)  #0.08204674012245144


################IMROVEMENT######################
from statistics import mean
Eigenfaces1_all, Eigenfaces2_all=eigenfaces(3)

mean_Eigen_vect_sub_1=pd.DataFrame(np.asarray(Eigenfaces1_all)).mean(axis=0)
mean_Eigen_vect_sub_2=pd.DataFrame(np.asarray(Eigenfaces2_all)).mean(axis=0)

plt.figure()
plt.imshow(np.reshape(np.asarray(mean_Eigen_vect_sub_1),(61,80)),cmap="gray")
plt.show()
plt.figure()
plt.imshow(np.reshape(np.asarray(mean_Eigen_vect_sub_2),(61,80)),cmap="gray")
plt.show()


s1_1_mean=[]
s2_2_mean=[]
s2_1_mean=[]
s1_2_mean=[]
for i in range(0,3):
    s1_1_mean.append(scoresub(np.asarray(Eigenfaces1_all[i]), Reduced_image_Subject1))
    s2_2_mean.append(scoresub(np.asarray(Eigenfaces2_all[i]), Reduced_image_Subject2))
    s2_1_mean.append(scoresub(np.asarray(Eigenfaces2_all[i]), Reduced_image_Subject1))
    s1_2_mean.append(scoresub(np.asarray(Eigenfaces1_all[i]), Reduced_image_Subject2))
    
S1_1_mean=mean(s1_1_mean)
S2_2_mean=mean(s2_2_mean)
S2_1_mean=mean(s2_1_mean)
S1_2_mean=mean(s1_2_mean)








