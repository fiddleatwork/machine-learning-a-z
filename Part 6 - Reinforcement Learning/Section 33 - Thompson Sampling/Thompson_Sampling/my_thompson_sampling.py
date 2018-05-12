# -*- coding: utf-8 -*-
"""
Created on Thu May  3 21:29:13 2018

@author: michael
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
import random

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

ROUNDS = 10000
ADS = 10

ads_selected = []
total_reward = 0

numbers_of_rewards_1 = [0] * ADS
numbers_of_rewards_0 = [0] * ADS

for n in range(0, ROUNDS):
    ad = 0
    max_random = 0
    for i in range(0, ADS): 
        random_beta = random.betavariate(numbers_of_rewards_1[i] + 1, numbers_of_rewards_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n,ad]
    if reward == 1:
        numbers_of_rewards_1[ad] = numbers_of_rewards_1[ad] + 1
    else:
        numbers_of_rewards_0[ad] = numbers_of_rewards_0[ad] + 1
    total_reward = total_reward + reward
    
plt.hist(ads_selected)
plt.title('Histogram of ads selections')
plt.xlabel('Ads')
plt.ylabel('Number of times each ad was selected')
plt.show()
    