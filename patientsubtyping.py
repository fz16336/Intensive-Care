# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:48:03 2018

@author: Farrel
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

feature_dataset = pd.read_csv('data/Features.csv')
X_train = feature_dataset.iloc[:, 1:39].values
X_train_untouched = feature_dataset.iloc[:, 1:39].values
X = feature_dataset.iloc[:, 0:45].values

from sklearn.cluster import KMeans

# Variable for choice of number of cluster. 2 and 5 are the chosen number of clusters, 
# refer to ClusterEvaluation.py for the code, and the report for reasoning.
num_cluster = 5

fig = plt.figure()

kmeans = KMeans(n_clusters = num_cluster, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
y_kmeans = kmeans.fit_predict(X_train)
centroids = kmeans.fit(X_train).cluster_centers_
labels = kmeans.fit(X_train).labels_

# subtype assignment denoted as Subtype S1,S2,S3,S4,S5, *Note if num_cluster = 2
# Then S3,S4,S5 will just be empty
S1 = X[np.where(labels == 0)]
S2 = X[np.where(labels == 1)]
S3 = X[np.where(labels == 2)]
S4 = X[np.where(labels == 3)]
S5 = X[np.where(labels == 4)]

c0 = X_train[np.where(labels == 0)]
c1 = X_train[np.where(labels == 1)]

from sklearn.preprocessing import normalize
X_train = normalize(X_train)

# n_components determine number of principal axes, choose 3 since it gives a better overall
# variance percentage captured, refer to explain_variance variable
from sklearn.decomposition import PCA
pca = PCA(n_components = 3)
X_train = pca.fit_transform(X_train)
explain_variance = pca.explained_variance_ratio_


bx = fig.add_subplot(111, projection= '3d')
n = num_cluster 

#'''
for i in range(0, n):
    #colormap = plt.cm.gist_ncar
    xs = X_train[y_kmeans == 0+i, 0]
    ys = X_train[y_kmeans == 0+i, 1]
    zs = X_train[y_kmeans == 0+i, 2]
    bx.scatter(xs,ys,zs, label ='Patient subtype %d' %i, alpha = 0.5)
#'''
'''
c0 = X_train[np.where(labels == 0)]
c1 = X_train[np.where(labels == 1)]
#c2 = X_train[np.where(labels == 2)]
#c3 = X_train[np.where(labels == 3)]
#c4 = X_train[np.where(labels == 4)]

bx.scatter(c0[:, 0], c0[:, 1], c0[:, 2], s=120, alpha = 0.5, label='Patients in cluster 1')
bx.scatter(c1[:, 0], c1[:, 1], c1[:, 2], s=120, alpha = 0.5, label='Patients in cluster 2')
#bx.scatter(c2[:, 0], c2[:, 1], c2[:, 2], s=120, alpha = 0.5, label='Patients in cluster 3')
#bx.scatter(c3[:, 0], c3[:, 1], c3[:, 2], s=120, alpha = 0.5, label='Patients in cluster 4')
#bx.scatter(c4[:, 0], c4[:, 1], c4[:, 2], s=120, alpha = 0.5, label='Patients in cluster 5')
'''
   
f = 15
bx.set_xlabel('PC1', fontsize = f)
bx.set_ylabel('PC2', fontsize = f)
bx.set_zlabel('PC3', fontsize = f)

bx.legend(loc='best', fontsize = 'xx-large')