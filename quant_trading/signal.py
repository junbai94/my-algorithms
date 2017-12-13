# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:51:41 2017

@author: junbai

Signal and position
"""

import pandas as pd
import numpy as np

def comparison_signal(df, px, long_signal, short_signal):
    long_entry = long_signal[0]
    long_exit = long_signal[1]
    short_entry = short_signal[0]
    short_exit = short_signal[1]
    
    df['num unit long'] = np.nan
    df['long_entry'] = ((df[px] < df[long_entry]) & (df[px].shift() >= df[long_entry].shift()))
    df['long_exit'] = ((df[px] >= df[long_exit]) & (df[px].shift() < df[long_exit].shift()))    
    df.loc[df['long_entry'],'num units long'] = 1 
    df.loc[df['long_exit'],'num units long'] = 0 
    df['num units long'][0] = 0 
    df['num units long'] = df['num units long'].fillna(method='pad') 
    
    df['num unit short'] = np.nan
    df['short_entry'] = ((df[px]>df[short_entry]) & (df[px].shift() <= df[short_entry].shift()))
    df['short_exit'] = ((df[px] <= df[short_exit]) & (df[px].shift() >= df[short_entry].shift()))
    df.loc[df['short_entry'],'num units short'] = -1
    df.loc[df['short_exit'],'num units short'] = 0
    df['num units short'][0] = 0
    df['num units short'] = df['num units short'].fillna(method='pad')
    
    df['numUnits'] = df['num units long'] + df['num units short']
    return df

def daily_pnl(df, initial_wealth = 1000000):
    pass