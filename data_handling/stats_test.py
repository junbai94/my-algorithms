# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:04:54 2017

@author: junbai

This script summarizes some statistical tests
"""

from statsmodels.tsa.stattools import coint, adfuller
import numpy as np
from numpy import log, polyfit, sqrt, std, subtract
import pandas as pd
import statsmodels.tsa.stattools as ts
import statsmodels.api
import matplotlib.pyplot as plt
from pandas.plotting import autocorrelation_plot
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 

def test_stationary(X, threshold=0.01):
    """
    Test if a time series is stationary
    Pre-condition:
        X - a pandas Series
    """
    pvalue = adfuller(X)[1]
    if pvalue < threshold:
        print 'p-value = ' + str(pvalue) + ' The series is likely stationary.'
        return True
    else:
        print 'p-value = ' + str(pvalue) + ' The series is likely non-stationary.'
        return False

def test_mean_reverting(X):
    """
    Test if a time series is mean reverting
    """
    Y = X.copy(False)
    cadf = adfuller(Y)
    print 
    print 'Augmented Dickey Fuller test statistic =',cadf[0]
    print 'Augmented Dickey Fuller p-value =',cadf[1]
    print 'Augmented Dickey Fuller 1%, 5% and 10% test statistics =',cadf[4]
    
def hurst(X):
	"""Returns the Hurst Exponent of the time series vector ts"""
	# Create the range of lag values
	lags = range(2, 100)
 
	# Calculate the array of the variances of the lagged differences
	tau = [sqrt(std(subtract(X[lag:], X[:-lag]))) for lag in lags]
 
	# Use a linear fit to estimate the Hurst Exponent
	poly = polyfit(log(lags), log(tau), 1)
 
	# Return the Hurst exponent from the polyfit output
	return poly[0]*2.0

def half_life(series):
    """
    Caculate half life of a mean reverting time series
    Pre-condition:
        series - a mean reverting pandas series
    """
    # re-initialize series's index
    X = series.copy(False)
    X.index = range(len(X))
    #Run OLS regression on spread series and lagged version of itself
    spread_lag = X.shift(1)
    spread_lag.ix[0] = spread_lag.ix[1]
    spread_ret = X - spread_lag
    spread_ret.ix[0] = spread_ret.ix[1]
    spread_lag2 = statsmodels.api.add_constant(spread_lag)
     
    model = statsmodels.api.OLS(spread_ret,spread_lag2)
    res = model.fit()
     
     
    halflife = round(-np.log(2) / res.params[1],0)
     
    if halflife <= 0:
        halflife = 1
    
    return halflife

def autocorrelation_graph(X):
    """
    Plot the autocorrelation graph of a pandas series
    Pre-condition:
        X - a pandas Series
    """
    plt.figure(figsize=(10, 5))
    autocorrelation_plot(X)
    plt.show()
    
def price_seasonality(data):
    data = data.copy()
    px = data.columns[1]
    start_year = data['date'][0].year
    end_year = data['date'].iloc[-1].year
    num_years = end_year - start_year + 1
    
    for i in range(num_years):
        temp = data[data['date'].dt.year==start_year+i]
#        temp[px] = pd.ewma(temp[px], span=10)
        temp['trend'] = pd.ewma(temp[px], span=250)
        plt.plot(temp['date'], temp[px] - temp['trend'])
        plt.title("Seasonality plot of year {}".format(start_year+i))
        plt.xticks(rotation="vertical")
        plt.show()
        

def vol_seasonality(data):
    data = data.copy()
    px = data.columns[1]
    start_year = data['date'][0].year
    end_year = data['date'].iloc[-1].year
    num_years = end_year - start_year + 1
    
    for i in range(num_years):
        temp = data[data['date'].dt.year==start_year+i]
        temp['log'] = np.log(temp[px])
        temp['log_ret'] = temp['log'] - temp['log'].shift()
        plt.plot(temp['date'], pd.rolling_std(temp['log_ret'], 10))
        plt.title("Volatiliy seasonality plot of year {}".format(start_year+i))
        plt.xticks(rotation="vertical")
        plt.show()

if __name__ == '__main__':
    pass