# -*- coding: utf-8 -*-
"""
Created on Thu Nov 30 16:57:59 2017

@author: junbai

Rank stocks based on momentum slope
"""

import numpy as np  # we're using this for various math operations
from scipy import stats  # using this for the reg slope
import pandas as pd
import sqlite3
from dbaccess import get_stock_data


def slope(ts):
    """
    Input: Price time series.
    Output: Annualized exponential regression slope, multipl
    """
    x = np.arange(len(ts))
    log_ts = np.log(ts)
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_ts)
    annualized_slope = (np.power(np.exp(slope), 250) - 1) * 100
    return annualized_slope * (r_value ** 2) 


def inv_vola_calc(ts):
    """
    Input: Price time series.
    Output: Inverse exponential moving average standard deviation. 
    Purpose: Provides inverse vola for use in vola parity position sizing.
    """
    returns = np.log(ts).diff()
    stddev = returns.ewm(halflife=20, ignore_na=True, min_periods=0,
                         adjust=True).std(bias=False).dropna()
    return 1 / stddev.iloc[-1]


def main(code_list):
    
    momentum_window = 60  # first momentum window.
    momentum_window2 = 90  # second momentum window
    
    # Limit minimum slope. Keep in mind that shorter momentum windows
    # yield more extreme slope numbers. 
    minimum_momentum = 60  # momentum score cap
    
    number_of_stocks = 5  # portfolio size
    
    index_average_window = 100  # moving average periods for index filter
    
    exclude_days = 5  # excludes most recent days from momentum calculation
    
    data_end = -1 * exclude_days # exclude most recent data
    momentum1_start = -1 * (momentum_window + exclude_days)
    momentum2_start = -1 * (momentum_window2 + exclude_days)
    
    result = pd.Series()
    
    for code in code_list:
        data = get_stock_data(code, frm='2016-05-01')
        momentum_hist1 = data[momentum1_start:data_end]['Close']
        momentum_hist2 = data[momentum2_start:data_end]['Close']
        
        slope1 = slope(momentum_hist1)
        slope2 = slope(momentum_hist2)
        mean = (slope1 + slope2) / 2.0
        result = result.append(pd.Series(mean, index=[code,]))
        
        
    
    return result