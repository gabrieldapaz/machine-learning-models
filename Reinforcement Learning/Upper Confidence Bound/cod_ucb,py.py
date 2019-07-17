# -*- coding: utf-8 -*-

#%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

df = pd.read_csv('Ads_CTR_Optimisation.csv')

N = 10000 #numero de rounds do dataframe
d = 10 #numero de anuncios do dataframe

ads_selected = [] #vetor de tamanho N dizendo em cada round qual ads foi selecionado
numbers_of_selections = [0] * d #vetor que contem quantas vezes cada anuncio foi selecionado
sums_of_rewards = [0] * d #vetor com a soma das recompensas para cada anuncio
total_reward  = 0

for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if (numbers_of_selections[i] > 0):
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = df.values[n, ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward

plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
            