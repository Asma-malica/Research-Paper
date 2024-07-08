# GWT-RF-ATT
We provide the code for the GWT-RF-ATT method(https://doi.org/10.1016/j.eswa.2022.117483). Please follow the instructions below to use our code.
## Prerequisites
The code is tested on 64 bit Windows 10. You should also install Python 3.8 before running our code.
## motor.rar and total.rar files
The implementation of the GWT-RF-ATT method mainly includes the following steps:

cluster;

Calculate the distance between features;

Find the shortest and longest distance between features;

Calculate the wavelet coefficients of the feature vector;

Use RF to predict patient severity;

Improving RF using attention mechanism.

Therefore, the motor and total folders contain the following files:
## 1）cluster.py
Clustering patient data. This paper divides the data into 4 clusters。
## 2）0 Calculate distance between features.py
Judging similarity between features by Euclidean distance
## 3）1 Shortest Path_Wavelet Coefficients_RF Prediction.py
Find the shortest path between features, perform wavelet transform on the feature vector, and use RF to predict the severity of the disease
## 4）1 Longest Path_Wavelet Coefficients_RF Prediction.py
Find the shortest path between features, perform wavelet transform on the feature vector, and use RF to predict the severity of the disease
## 5）1 Original Path_Wavelet Coefficients_RF Prediction.py
Wavelet transform the original feature vector and use RF to predict disease severity
## 6）2 Improving RF with Attention Mechanism.py
The frequency features are input into the attention weighted RF to predict the severity of PD, allowing the results of decision trees with better predictive performance in the RF to be highlighted while reducing the risk of overfitting.
