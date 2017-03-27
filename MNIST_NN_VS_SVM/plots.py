import pandas as pd
import sklearn
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import numpy as np
import os, struct
from array import array as pyarray
from numpy import append, array, int8, uint8, zeros
from pylab import *
from numpy import *
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

param_range = [120,100,80,60,40,35,30,20]


train_scores_mean1 = [1,1,1,1,1]  #1 => learning rates
train_scores_mean2 = [0.9706,0.9889,0.9962,0.9974,0.9987] #2 => C values
train_scores_mean3 = [1,1,1,1,1] #3 => Momentum values
train_scores_mean4 = [1,1,1,1,1] #4 => Batch size values
train_scores_mean5 = [1,1,1,1,1] #5 => hidden nodes size values
test_scores_mean1 = [0.9639,0.9667,0.9678,0.9681,0.9708]
test_scores_mean2 = [0.9302,0.9209,0.9137,0.9126,0.9111]
test_scores_mean3 = [0.965,0.9665,0.9679,0.9659,0.0964] 
test_scores_mean4 = [0.9689,0.9704,0.9697,0.9656,0.9657]
test_scores_mean5 = [0.9509,0.9593,0.9618,0.9631,0.966]

pca_values = [0.9468, 0.9522, 0.9558, 0.9602, 0.9608, 0.962, 0.9622, 0.9616]

plt.title("Testing Curve for SVC using PCA")
plt.xlabel("Number of components")
plt.ylabel("Score")
plt.ylim(0.94, 0.975)
plt.plot(param_range, pca_values, label="Testing score",
             color="navy")
#plt.plot(param_range, test_scores_mean4, label="Cross-validation score",
#             color="navy")

plt.legend(loc="best")
plt.show()