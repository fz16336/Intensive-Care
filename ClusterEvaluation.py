# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 14:48:03 2018

@author: Farrel
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans


feature_dataset = pd.read_csv('data/Features.csv')
X_train = feature_dataset.iloc[:, 0:38].values
X = feature_dataset.iloc[:, 0:38].values

# Within cluster sum of square distances.
ssd = []

upperbound = 9
for i in range(1,upperbound):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(X_train)
    ssd.append(kmeans.inertia_)

f = 33
plt.plot(range(1,upperbound), ssd, linewidth = 5, alpha = 1, color ='r')
plt.title('Validation for value of K via Elbow Method', fontsize = f)
plt.xlabel('Number of Clusters', fontsize = f)
plt.xticks(fontsize = f - 5)
plt.yticks(fontsize = f - 5)
plt.xlim(1,upperbound - 1)
plt.ylabel('SSD', fontsize = f)
plt.grid(True)
plt.show()