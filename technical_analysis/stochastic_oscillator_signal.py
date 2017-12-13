# -*- coding: utf-8 -*-
"""
Created on Mon Nov 13 17:58:39 2017

@author: junbai

Generate signal on Stochastic Oscillator KD 
"""

from technical_analysis import *
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.dates import DateFormatter, WeekdayLocator, DayLocator, MONDAY, date2num
from matplotlib.finance import candlestick_ohlc

###############################################################################

# Apple Stock for testing
aapl = pd.read_csv("C:/Users/j291414/Desktop/AAPL.csv")
aapl['Date'] = pd.to_datetime(aapl['Date'], format="%Y-%m-%d")
aapl.index = aapl['Date']

###############################################################################

plt.figure(1)
plt.subplot(211)
plt.plot(aapl['Date'], aapl['Adj Close'])

plt.subplot(212)
SOk = FULL_STOK(aapl, 9, 3)['SO%k']
plt.plot(aapl['Date'], SOk, 'b')
plt.plot(aapl['Date'], 20*np.ones(len(SOk)), 'r--')
plt.plot(aapl['Date'], 80*np.ones(len(SOk)), 'r--')
plt.show()


plt.show()

###############################################################################

#import matplotlib.pyplot as plt
#import numpy as np
#
#t = np.arange(0.01, 5.0, 0.01)
#s1 = np.sin(2*np.pi*t)
#s2 = np.exp(-t)
#s3 = np.sin(4*np.pi*t)
#
#ax1 = plt.subplot(311)
#plt.plot(t, s1)
#plt.setp(ax1.get_xticklabels(), fontsize=6)
#
## share x only
#ax2 = plt.subplot(312, sharex=ax1)
#plt.plot(t, s2)
## make these tick labels invisible
#plt.setp(ax2.get_xticklabels(), visible=False)
#
## share x and y
#ax3 = plt.subplot(313, sharex=ax1, sharey=ax1)
#plt.plot(t, s3)
#plt.xlim(0.01, 5.0)
#plt.show()