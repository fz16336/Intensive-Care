# Intensive-Care
Mathematical and Data Modelling project on recognising patterns in the internsive care using Unsupervised Learning algorithms.
This repository contains the report of findings and statistics inferred from analysing the PhysioNet Challenge dataset, as well as the Python codes used for clustering and statistical analysis. 

## Dataset
The dataset was given from the PhysionNet Challenge. It consist of 4000 rows of anonymised patient ID with 70 columns (38 after data-preprocessing and feature engineering) health charectristics features, as well as a seperate dataset for discharge metrics to infer the patients health outcome after admission. 

## Contents
* PatientSubtyping.py: Clustering assignment for the subtypes (clusters) present in the dataset, using K-means clustering and PCA.
* Evaluation.py: Elbow method to confirm number of k-clusters.
* Statistics folder:
  * FeatureStatistics.py: Statistics on the distribution of features present in the main dataset
  * ClusterStatistics.py: Statistical analysis of each cluster for both k=2 and k=5
  * SurvivalStatistics.py: Statistics of patients well-being after being discharge from intensive care
  * Statistics.ipynb:Jupyter notebook of all the statistical analysis of the 3 .py files combined for an easy read evaluation
  
## Summary of Result
Aspartate transaminase (AST) and Alanine transminase (ALT) are enzymes present in the liver which becomes the basis of our findings as both of them totaled to capturing around 90% of the variances in the data. They also become the main contribution in terms of the seperation between the two clusters. Cluster analyis were done by comparing k=2 clusters, based on evaluation by the Elbow method, and k=5 clusters (too see if there are any other subpopulations inside the two main clusters).
  
![k=2](/Intensive-Care/figures/k=2.png)
![k=5](/Intensive-Care/figures/k=5.png)
  


 
