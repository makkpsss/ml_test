# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 11:16:05 2018

@author: user
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values


from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

y_pred = regressor.predict(7)


x_grid = np.arrange()