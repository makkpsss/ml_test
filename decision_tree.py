# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 15:38:28 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("beacon_dataset.csv")
bk_dataset = dataset
X = dataset.iloc[:, 1:4].values
y = dataset.iloc[:,5].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

labelEncoder_X = LabelEncoder()
X[:, 1] = labelEncoder_X.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features = [1])

X = onehotencoder.fit_transform(X).toarray()
