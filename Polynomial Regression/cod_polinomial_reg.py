import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("Position_Salaries.csv")
X = df.iloc[:, 1:2].values
y = df.iloc[:, 2].values

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X,y)

"""nessa parte PolynomialFeatures é uma ferramenta
usada para transformar uma matriz em forma polynomial, ou seja,
criando os fatores ao quadrado.
Feito isso, voce cria outro objeto linear com a matrix polynomial
"""

from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y) 
regressor_2 = LinearRegression()
regressor_2.fit(X_poly, y)

plt.scatter(X,y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.show()

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor_2.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.show()

regressor.predict([[6.5]])

regressor_2.predict(poly_reg.fit_transform([[6.5]]))