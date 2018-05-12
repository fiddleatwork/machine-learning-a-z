# -*- coding: utf-8 -*-
"""
Created on Thu May  3 21:29:13 2018

@author: michael
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math

dataset = pd.read_csv("Ads_CTR_Optimisation.csv")

ROUNDS = 10000
ADS = 10

numbers_of_selections = [0] * ADS
sums_of_rewards = [0] * ADS
ads_selected = []
total_reward = 0
for n in range(0, ROUNDS):
    max_upper_bound = 0
    ad = 0
    for i in range(0, ADS): 
        if numbers_of_selections[i] > 0:
            average_reward = sums_of_rewards[i] / numbers_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n+1) / numbers_of_selections[i])
            upper_bound = average_reward + delta_i
        else : 
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    numbers_of_selections[ad] = numbers_of_selections[ad] + 1
    reward = dataset.values[n,ad]
    sums_of_rewards[ad] = sums_of_rewards[ad] + reward
    total_reward = total_reward + reward
    
    