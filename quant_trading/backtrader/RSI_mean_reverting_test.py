# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 14:41:52 2017

@author: j291414
"""
import sys
sys.path.append("C:/Users/j291414/my algorithm")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import technical_analysis.technical_analysis as ta
from data_handling.stats_test import *

data = pd.read_csv("C:/Users/j291414/Desktop/sample_data.csv")
data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
data = ta.RSI(data, 21)
#plt.figure(figsize=(20, 10))
#plt.plot(data['Date'], data['RSI_21'])
#plt.show()

target = data['RSI_21'].dropna()
test_stationary(target)
test_mean_reverting(target)
half_life(target)