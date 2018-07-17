# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 18:19:56 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 3].values

#from sklearn.preprocessing import Imputer
#imputer = Imputer(missing_values="NaN", strategy='mean', axis=0)
#imputer.fit(X[:, 1:3])
#X[:, 1:3] = imputer.transform(X[:, 1:3])


#from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelEncoder_X = LabelEncoder()
#X[:,0] = labelEncoder_X.fit_transform(X[:, 0])
#onehotencoder = OneHotEncoder(categorical_features=[0])
#X = onehotencoder.fit_transform(X).toarray()
#labelEncoder_Y = LabelEncoder()
#y = labelEncoder_Y.fit_transform(y)

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=0)

"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""
