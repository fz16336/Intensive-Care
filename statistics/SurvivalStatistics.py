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

# Statistics for various discharge metric data
labels = ["SAPS-I", "SOFA", "Survival Rate After Discharge", "In Hospital Length of Stay"]
ylabels = ["SAPS-I", "SOFA", "Mean values", "Average days after discharge"]

#set k = 42 for length of stay and k=41 for survival rate
k = 42

subtype0_dataset = pd.read_csv('data/pdata05.csv')
s0 = subtype0_dataset.iloc[:, k].values
s0[np.isnan(s0)] = 0

subtype1_dataset = pd.read_csv('data/pdata15.csv')
s1 = subtype1_dataset.iloc[:, k].values
s1[np.isnan(s1)] = 0

subtype2_dataset = pd.read_csv('data/pdata25.csv')
s2 = subtype2_dataset.iloc[:, k].values
s2[np.isnan(s2)] = 0

subtype3_dataset = pd.read_csv('data/pdata35.csv')
s3 = subtype3_dataset.iloc[:, k].values
s3[np.isnan(s3)] = 0

subtype4_dataset = pd.read_csv('data/pdata45.csv')
s4 = subtype4_dataset.iloc[:, k].values
s4[np.isnan(s4)] = 0

sm0 = np.mean(s0)
sm1 = np.mean(s1)
sm2 = np.mean(s2)
sm3 = np.mean(s3)
sm4 = np.mean(s4)

if k == 39:
    label1 = labels[0]
    label2 = ylabels[0]
elif k == 40:
    label1 = labels[1]
    label2 = ylabels[1]
elif k == 41:
    label1 = labels[2]
    label2 = ylabels[2]
elif k == 42:
    label1 = labels[3]
    label2 = ylabels[3]

plt.bar(["Subtype 0","Subtype 1","Subtype 2","Subtype 3","Subtype 4"],
        [sm0,sm1,sm2,sm3,sm4],
        color = ['royalblue', 'gold', 'mediumspringgreen','firebrick', 'mediumorchid'],
        edgecolor = 'black',
        alpha = 0.7)

plt.title('Average ' + label1, fontsize = 20)
plt.xticks(fontsize=20, rotation=45)
plt.yticks(fontsize=20)
plt.ylabel(label2, fontsize = 20)
plt.show()

# Comparing cluster survival statistics with overall population statistics
patient_dataset = pd.read_csv('data/patients_data.csv')
p = patient_dataset.iloc[:, k]
p.loc[np.isnan(p)] = 0
mp = np.mean(p)
sp = np.std(p)

print("Subtype's " + label1 + " mean: ", [sm0,sm1,sm2,sm3,sm4])
print("Population " + label1 + " mean: ", mp)

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
# Statistics on mortality rate
k = 43
subtype0_dataset = pd.read_csv('data/pdata05.csv')
s0 = subtype0_dataset.iloc[:, k].values
s0[np.isnan(s0)] = 0

subtype1_dataset = pd.read_csv('data/pdata15.csv')
s1 = subtype1_dataset.iloc[:, k].values
s1[np.isnan(s1)] = 0

subtype2_dataset = pd.read_csv('data/pdata25.csv')
s2 = subtype2_dataset.iloc[:, k].values
s2[np.isnan(s2)] = 0

subtype3_dataset = pd.read_csv('data/pdata35.csv')
s3 = subtype3_dataset.iloc[:, k].values
s3[np.isnan(s3)] = 0

subtype4_dataset = pd.read_csv('data/pdata45.csv')
s4 = subtype4_dataset.iloc[:, k].values
s4[np.isnan(s4)] = 0

deaths0 = []
lives0 = []
for i in range(len(s0)):
    if s0[i] == 1:
        d = s0[i]
        deaths0.append(d)
    elif s0[i] == 0:
        l = s0[i]
        lives0.append(l)

deaths1 = []
lives1 = []
for i in range(len(s1)):
    if s1[i] == 1:
        d = s1[i]
        deaths1.append(d)
    elif s1[i] == 0:
        l = s1[i]
        lives1.append(l)

deaths2 = []
lives2 = []
for i in range(len(s2)):
    if s2[i] == 1:
        d = s2[i]
        deaths2.append(d)
    elif s2[i] == 0:
        l = s2[i]
        lives2.append(l)

deaths3 = []
lives3 = []
for i in range(len(s3)):
    if s3[i] == 1:
        d = s3[i]
        deaths3.append(d)
    elif s3[i] == 0:
        l = s3[i]
        lives3.append(l)

deaths4 = []
lives4 = []
for i in range(len(s4)):
    if s4[i] == 1:
        d = s4[i]
        deaths4.append(d)
    elif s4[i] == 0:
        l = s4[i]
        lives4.append(l)

#print(s0)
#print(len(deaths0), len(lives0))
#print(len(deaths0) + len(lives0))

plt.bar(["Subtype0", "Subtype1", "Subtype2", "Subtype3", "Subtype4"],
        [len(deaths0)/len(s0), len(deaths1)/len(s1), len(deaths2)/len(s2), len(deaths3)/len(s3), len(deaths4)/len(s4)],
        color = ['royalblue', 'gold', 'mediumspringgreen','firebrick', 'mediumorchid'],
        alpha = 0.7)

plt.title("Proportion of In-hospital Deaths", fontsize = 20)
plt.xticks(fontsize = 20, rotation = 45)
plt.ylabel("Percentage", fontsize = 20)
plt.grid(True)
plt.show()

print("Percentages of patients dying in admission or cluster")
print('Subtype0: '+ str(len(deaths0)/len(s0)*100) + '%')
print('Subtype0: '+ str(len(deaths1)/len(s1)*100) + '%')
print('Subtype0: '+ str(len(deaths2)/len(s2)*100) + '%')
print('Subtype0: '+ str(len(deaths3)/len(s3)*100) + '%')
print('Subtype0: '+ str(len(deaths4)/len(s4)*100) + '%')
