# -*- coding: utf-8 -*-
"""
Created on Mon Nov 06 09:10:31 2017

@author: junbai

Support and Resistance Level
"""

import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
import numpy as np
import trendy

###############################################################################

conn = sqlite3.connect("C:/Users/j291414/Desktop/market_data.db")
sql = "select * from fut_daily where instID = 'i1801'"
df = pd.read_sql_query(sql, conn)
df = df[['date', 'close']]
df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d %H:%M:%S')

###############################################################################

num = len(df['close'])
plt.plot(df['date'], df['close'])
plt.plot(df['date'], 570*np.ones([num, 1]))
plt.plot(df['date'], 631.5*np.ones([num, 1]))
plt.plot(df['date'], 435.5*np.ones([num, 1]))
plt.xticks(rotation='vertical')
plt.show()

###############################################################################

def supres(ltp, n):
    """
    This function takes a numpy array of last traded price
    and returns a list of support and resistance levels 
    respectively. n is the number of entries to be scanned.
    """
    from scipy.signal import savgol_filter as smooth

    # converting n to a nearest even number
    if n % 2 != 0:
        n += 1

    n_ltp = ltp.shape[0]

    # smoothening the curve
    ltp_s = smooth(ltp, (n + 1), 3)

    # taking a simple derivative
    ltp_d = np.zeros(n_ltp)
    ltp_d[1:] = np.subtract(ltp_s[1:], ltp_s[:-1])

    resistance = []
    support = []

    for i in xrange(n_ltp - n):
        arr_sl = ltp_d[i:(i + n)]
        first = arr_sl[:(n / 2)]  # first half
        last = arr_sl[(n / 2):]  # second half

        r_1 = np.sum(first > 0)
        r_2 = np.sum(last < 0)

        s_1 = np.sum(first < 0)
        s_2 = np.sum(last > 0)

        # local maxima detection
        if (r_1 == (n / 2)) and (r_2 == (n / 2)):
            resistance.append(ltp[i + ((n / 2) - 1)])

        # local minima detection
        if (s_1 == (n / 2)) and (s_2 == (n / 2)):
            support.append(ltp[i + ((n / 2) - 1)])

    return support, resistance

###############################################################################

# test supres function
npArray = df['close'].as_matrix()

support, resistance = supres(npArray, 5)

###############################################################################

# test trendy 
close = df['close']
close.index = df['date']

cutoff = '2000-08-01'
cut = close[close.index > cutoff]

trendy.gentrends(cut, window=1./3, charts=True)
trendy.segtrends(cut, segments=3, charts=True)
trendy.minitrends(cut, window = 30, charts = True)
trendy.iterlines(cut, window = 30, charts = True)
