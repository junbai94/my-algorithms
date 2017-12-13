# -*- coding: utf-8 -*-
"""
Created on Fri Nov 17 10:29:57 2017

@author: junbai

Signal and back-test testing script
"""
import sys
sys.path.append("C:/Users/j291414/my algorithms")

import sqlite3
import pandas as pd
import statsmodels.formula.api as sm
import matplotlib.pyplot as plt
import numpy as np
from technical_analysis.technical_analysis import *
from quant_trading.signal import *

###############################################################################

# import a stock as example
df = pd.read_csv("C:/Users/j291414/Desktop/sample_data.csv")
df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")

# select recent years
starting_year = '2015-01-01'
df = df[df['Date'] >= starting_year]
df.index = range(len(df))

# Bollinger Band on df
df = BBANDS(df, 20)

# Signal based on Bollinger Band
px = 'Close'
long_signal = ('Bollingerb_20', 'MA_20')
short_signal = ('BollingerB_20', 'MA_20')
df = comparison_signal(df, px, long_signal, short_signal)

###############################################################################
