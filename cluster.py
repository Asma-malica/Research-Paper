#!/usr/bin/python3
# -*- coding:utf-8 -*-
# import pylab as pl
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

def xy():
    parkinson_dataset = pd.read_csv("parkinson.csv")
    parkinson_dataset = parkinson_dataset.dropna()
    columns = ['motor_UPDRS', 'total_UPDRS']
    parkinson_x = parkinson_dataset.drop(columns, axis=1)
    #         # print(parkinson_x.shape)       # (5875, 18)
    columns = ['motor_UPDRS']
    parkinson_y = pd.DataFrame(parkinson_dataset, columns=columns)

    dataset = parkinson_x.values
    parkinson_x = dataset.astype('float32')
    dataset = parkinson_y.values
    parkinson_y = dataset.astype('float32')
    print(parkinson_y.shape)
    # X_train, X_test, y_train, y_test = train_test_split(parkinson_x, parkinson_y,test_size=0.2, random_state=67)
    # return X_train, X_test, y_train, y_test
    return parkinson_x,parkinson_y

parkinson_x,parkinson_y=xy()

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 4)
kmeans.fit(parkinson_x)
b=kmeans.labels_
b=b.tolist()
b=enumerate(b)

a00 = [] 
# x input y output --> cluster 
y00 = []
a01 = []
y01 = []
a02 = []
y02 = []
a03 = []
y03 = []

for index,value in b:

    if value==0:
        y00.append(parkinson_y[index, :])
        a00.append(parkinson_x[index,:])
    if value==1:
        y01.append(parkinson_y[index, :])
        a01.append(parkinson_x[index,:])
    if value==2:
        y02.append(parkinson_y[index, :])
        a02.append(parkinson_x[index,:])
    if value==3:
        y03.append(parkinson_y[index, :])
        a03.append(parkinson_x[index,:])

a00=np.array(a00)
print(a00.shape)
a01=np.array(a01)
print(a01.shape)
a02=np.array(a02)
print(a02.shape)
a03=np.array(a03)
print(a03.shape)

a00=pd.DataFrame(a00)
a00.to_csv("b00.csv")
y00 = pd.DataFrame(y00)
y00.to_csv("y00.csv")

a01=pd.DataFrame(a01)
a01.to_csv("b01.csv")
y01 = pd.DataFrame(y01)
y01.to_csv("y01.csv")

a02=pd.DataFrame(a02)
a02.to_csv("b02.csv")
y02 = pd.DataFrame(y02)
y02.to_csv("y02.csv")

a03=pd.DataFrame(a03)
a03.to_csv("aa.csv")
y03 = pd.DataFrame(a03)
y03.to_csv("aa00.csv")




















'''
选择k值 
K = range(1, 30)
meandistortions = []
for k in K:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(x)
    meandistortions.append(sum(np.min(cdist(x, kmeans.cluster_centers_,'euclidean'), axis=1)) / x.shape[0])#选择每行最小距离求和
plt.figure()
plt.grid(True)
plt1 = plt.subplot(2,1,1)
plt1.plot(x[:,0], x[:,1], 'k.')
plt2 = plt.subplot(2,1,2)
plt2.plot(K, meandistortions)
plt.show()
'''