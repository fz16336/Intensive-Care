# Intensive-Care
Mathematical and Data Modelling project on recognising patterns in the intensive care using Unsupervised Learning algorithms.
This repository contains the report of findings and statistics inferred from analysing the PhysioNet Challenge dataset, as well as the Python codes used for clustering and statistical analysis.

## Dataset
The dataset was taken from the PhysionNet Challenge. It consist of 4000 rows of anonymised patient ID with 70 columns (38 after data-pre-processing and feature engineering) health characteristics features, as well as a separate dataset for discharge metrics to infer the patients health outcome after admission.

## Contents
* PatientSubtyping.py: Clustering assignment for the subtypes (clusters) present in the dataset, using k-means clustering and PCA.
* Evaluation.py: Elbow method to confirm number of k-clusters.
* datacolecltion.m: Data pre-processing code
* Statistics folder:
  * FeatureStatistics.py: Statistics on the distribution of features present in the main dataset
  * ClusterStatistics.py: Statistical analysis of each cluster for both k=2 and k=5
  * SurvivalStatistics.py: Statistics of patients well-being after being discharge from intensive care
  * Statistics.ipynb:Jupyter notebook of all the statistical analysis of the 3 .py files combined for an easy read evaluation

## Summary of Result
*Aspartate transaminase* (*AST*) and *Alanine transaminase* (*ALT*) are enzymes present in the liver which becomes the basis of our findings as both of them totalled to capturing around 90% of the variances in the data. They also become the main contribution in terms of the separation between the two clusters where in k=2 clusters the clusters differ in terms of *AST* and *ALT* levels, with cluster subtype0 depicting abnormally high *AST* and *ALT* levels, in addition to being validly obese (commonly accepted causality). This were then solidified by comparing their survival statistics where cluster subtype0 possesses higher mortality rate as well. All in all, this shows signs of liver damage as a primary cause of their admission to the intensive care.

Cluster analysis were done by comparing k=2 clusters, based on evaluation by the Elbow method, and k=5 clusters (too see if there are any other subpopulation difference inside the two main clusters).

![clusters](/clusters.png)
<!-- ![$$k$$=5](/figures/$$k$$=5.png) -->

## Future Works
Currently attempting to incorporate 'time-aware' LSTM networks as an alternative/additional model.
