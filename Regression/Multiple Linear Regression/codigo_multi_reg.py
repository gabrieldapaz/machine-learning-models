# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('50_Startups.csv')
X = df.iloc[:, :-1].values
y = df.iloc[:, 4].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3]) #transforma atributos categoricos em numeros
onehotencoder = OneHotEncoder(categorical_features = [3]) #cria as dummy variables
X = onehotencoder.fit_transform(X).toarray()

X = X[:, 1:]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

pred_y = regressor.predict(X_test)

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50,1)), values = X, axis=1)

def backwardElimination(x, sl):
    num_atri = len(x[0])
    for i in range(0, num_atri):
        regressor_OLS = sm.OLS(y, x).fit()
        max_atri = max(regressor_OLS.pvalues).astype(float)
        if max_atri > sl:
            for j in range(0, num_atri - i):
                if (regressor_OLS.pvalues[j].astype(float) == max_atri):
                    x = np.delete(x, j, 1)
    regressor_OLS.summary()
    return x
 
SL = 0.05
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
X_Modeled = backwardElimination(X_opt, SL)


