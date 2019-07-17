 # -*- coding: utf-8 -*-

#%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

"""
Um detalhe eh que o algortimo do apriori precisa estar no formato de lista de lista,
então precisamos ajustar o dataframe
"""

df = pd.read_csv("Market_Basket_Optimisation.csv", header= None)

transactions = []
for i in range(0,7501):
    transactions.append([str(df.values[i,j]) for j in range (0,20)])

from apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift =3, min_length= 2)

results = list(rules)
output = []
for row in results:
    output.append(str(row.items))