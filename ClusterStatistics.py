# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 12:35:01 2018

@author: Farrel
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('seaborn')

feature_dataset = pd.read_csv('data/Features.csv')
labels = ["Age","Gender","Height","ICUType","Weight","Albumin","ALP","ALT","AST","Bilirubin","BUN","Cholesterol",
          "Creatinine","DiasABP","FiO2","GCS","Glucose","HCO3","HCT","HR","K","Lactate","Mg","MAP","Na",
          "NIDiasABP","NIMAP","NISysABP","PaCO2","PaO2","pH","Platelets","RespRate","SaO2","SysABP","Temp",
          "Urine","WBC"]

clabels = ["PatientsID","Age","Gender","Height","ICUType","Weight","Albumin","ALP","ALT","AST","Bilirubin","BUN","Cholesterol",
          "Creatinine","DiasABP","FiO2","GCS","Glucose","HCO3","HCT","HR","K","Lactate","Mg","MAP","Na",
          "NIDiasABP","NIMAP","NISysABP","PaCO2","PaO2","pH","Platelets","RespRate","SaO2","SysABP","Temp",
          "Urine","WBC"]

labels_dict = {key: l for l, key in enumerate(labels)}

choice = str(input('Which health features within the clusters you want to analyse?: '))
i = labels_dict[choice]

# Fixing index difference
i = labels_dict[choice] + 1
j = i - 1

# Health features distribution in k=2 clusters
cluster1K2_dataset = pd.read_csv('data/cluster1k2.csv')
s1 = cluster1K2_dataset.iloc[:, i].values

cluster2K2_dataset = pd.read_csv('data/cluster2k2.csv')
s2 = cluster2K2_dataset.iloc[:, i].values

plt.hist([s1,s2], bins = 'sturges', alpha = 0.8, density = True,
         label = ['Patient subtype 0','Patient subtype 1'],
        color = ['royalblue','mediumspringgreen'],
        edgecolor = 'k')

plt.yscale('log')
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.title("Frequency of " + str(clabels[i]) + ' in each Subtype CLuster', fontsize = 20)
plt.legend(loc = 'best', fontsize = 20)
plt.grid(True)
plt.show()

# Density estimation of feature distribution in k=2 clusters
sns.distplot(s1, hist=False, kde=True,
             color = 'royalblue', kde_kws={'linewidth': 2}, label='Patient subtype 0')
sns.distplot(s2, hist=False, kde=True,
             color = 'mediumspringgreen', kde_kws={'linewidth': 2}, label='Patient subtype 1')

plt.title("Kernel Density Estimation for Maximum Likelihood Distribution of " + str(clabels[i]), fontsize = 20)
plt.xlabel("Range of " + str(clabels[i]) + " level")
plt.legend(loc = 'best', fontsize = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
population_dataset = pd.read_csv('data/Features.csv')
P = population_dataset.iloc[:, j].values

cluster1_dataset = pd.read_csv('data/cluster1.csv')
c1 = cluster1_dataset.iloc[:, i].values

cluster2_dataset = pd.read_csv('data/cluster2.csv')
c2 = cluster2_dataset.iloc[:, i].values

cluster3_dataset = pd.read_csv('data/cluster3.csv')
c3 = cluster3_dataset.iloc[:, i].values

cluster4_dataset = pd.read_csv('data/cluster4.csv')
c4 = cluster4_dataset.iloc[:, i].values

cluster5_dataset = pd.read_csv('data/cluster5.csv')
c5 = cluster5_dataset.iloc[:, i].values

f = 20

plt.hist([c1,c2,c3,c4,c5], alpha = 0.6, density = True,
         label = ['Patient subtype 0','Patient subtype 1','Patient subtype 2','Patient subtype 3','Patient subtype 4'],
        color = ['royalblue','mediumspringgreen','firebrick', 'mediumorchid','gold'],
        edgecolor = 'k')

# Frequency histogram for k=5 clusters
plt.title("Frequency of " + str(clabels[i]) + ' in each Subtype CLuster', fontsize = 20)
plt.legend(loc = 'best', fontsize = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.yscale('log')
plt.show()

#Kernel Density Estimation for k=5 clusters
sns.distplot(c1, hist=False, kde=True,
             color = 'cornflowerblue', kde_kws={'linewidth': 2}, label='Patient subtype 0')
sns.distplot(c2, hist=False, kde=True,
             color = 'sandybrown', kde_kws={'linewidth': 2}, label='Patient subtype 1')
sns.distplot(c3, hist=False, kde=True,
             color = 'mediumspringgreen', kde_kws={'linewidth': 2}, label='Patient subtype 2')
sns.distplot(c4, hist=False, kde=True,
             color = 'red', kde_kws={'linewidth': 2}, label='Patient subtype 3')
sns.distplot(c5, hist=False, kde=True,
             color = 'mediumorchid', kde_kws={'linewidth': 2}, label='Patient subtype 4')

plt.title("Kernel Density Estimation for Maximum Likelihood Distribution of " + str(clabels[i]), fontsize = 20)
plt.xlabel("Range of " + str(clabels[i]) + " level")
plt.legend(loc = 'best', fontsize = 20)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.show()

# Comparing clusters statistics agains overal patient population statistics
cm1 = np.mean(c1)
cm2 = np.mean(c2)
cm3 = np.mean(c3)
cm4 = np.mean(c4)
cm5 = np.mean(c5)

totalmeans = np.array([cm1,cm2,cm3,cm4,cm5])
populationmeans = np.mean(totalmeans)

var1 = np.var(c1)
var2 = np.var(c2)
var3 = np.var(c3)
var4 = np.var(c4)
var5 = np.var(c5)

totalvariances = np.array([var1,var2,var3,var4,var5])
populationvar = np.var(totalvariances)

print("Average of each clusters' " + str(clabels[i]) +" : ", totalmeans)

print("Variance of each clusters' " + str(clabels[i]) +" : ", totalvariances)

m = np.mean(P)
print("Average of patients' " + str(clabels[i]) +" : ", m)
var = np.var(P)
print("Variance of patients' " + str(clabels[i]) +" : ", var)
s= np.std(P)
print("Standard deviation of patients' " + str(clabels[i]) +" : ", s)
