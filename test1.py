# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:48:03 2018

@author: Farrel
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

feature_dataset = pd.read_csv('Features.csv')
X_train = feature_dataset.iloc[:, 0:37].values

from sklearn.cluster import KMeans

num_cluster = 5

fig = plt.figure()

ssd = []
for i in range(2,6):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X_train)
    ssd.append(kmeans.inertia_)

plt.plot(range(2,6), ssd)
plt.xlabel('Number of Clusters')
plt.ylabel('SSD')
plt.grid(True)
plt.show()

kmeans = KMeans(n_clusters = num_cluster, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X_train)

from sklearn.preprocessing import normalize
X_train = normalize(X_train)

from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
explain_variance = pca.explained_variance_ratio_


bx = fig.add_subplot(111, projection = '3d')
n = num_cluster
for i in range(0, n + 1):
    #colormap = plt.cm.gist_ncar
    xs = X_train[y_kmeans == 0+i, 0]
    ys = X_train[y_kmeans == 0+i, 1]
    zs = X_train[y_kmeans == 0+i, 2]
    bx.scatter(xs,ys,zs, label ='Cluster %d' %n, alpha = 0.5)
    
f = 15
bx.set_xlabel('PC1', fontsize = f)
bx.set_ylabel('PC2', fontsize = f)
bx.set_zlabel('PC3', fontsize = f)



