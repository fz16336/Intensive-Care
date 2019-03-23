# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 12:35:01 2018

@author: Farrel
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
plt.style.use('seaborn')
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

#
feature_dataset = pd.read_csv('data/Features.csv')

'''
After feature selection and data pre-processing, these are the available 38 health features available.
To see some statistics on a specific health feature, please choose one of them.
'''
labels = ["Age","Gender","Height","ICUType","Weight","Albumin","ALP","ALT","AST","Bilirubin","BUN","Cholesterol",
          "Creatinine","DiasABP","FiO2","GCS","Glucose","HCO3","HCT","HR","K","Lactate","Mg","MAP","Na",
          "NIDiasABP","NIMAP","NISysABP","PaCO2","PaO2","pH","Platelets","RespRate","SaO2","SysABP","Temp",
          "Urine","WBC"]

#feature_dataset.head()
#labels_dict = {key: l for l, key in enumerate(labels)}

# Choose of the 38 healths features to analys, e.g type ALT. *Note: ALT and AST are 
# the interesthing health features to see (refer to report).
choice = str(input('Which health features you want to analyse?: '))
i = labels_dict[choice]
D = feature_dataset.iloc[:, (i)].values

patientid = np.array([x for x in range(len(D))])

# Frequency histogram
plt.hist(D, bins='auto', density = False, edgecolor='k') #use fd, auto, and sturges
plt.yscale('log')
plt.title("Frequency of Patients' " + str(labels[(i)]))
plt.grid(True)
plt.show()

# Estimated distribution
sx = sns.distplot(D, hist= True, kde=True, norm_hist = True,
             color = 'cornflowerblue', kde_kws={'linewidth': 2})
sx.axes.set_title("Density Estimation of " + str(labels[(i)]))
sx.set_xlabel(str(labels[(i)]) + " Level")
plt.show()