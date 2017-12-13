# -*- coding: utf-8 -*-
"""
Created on Mon Oct 23 17:05:04 2017

@author: junbai

This script is for marking future curve. Likely to migrate to website
"""



'''
Spread between two years, like Cal17/Cal18 should be handled seperately.
Coz this matrix method can only handle vectors of two elements.
'''

import numpy as np
from numpy.linalg import inv
import pandas as pd
import sqlite3

conn = sqlite3.connect('C:/Users/j291414/Desktop/market_data.db')

TICK_SYMBOL = 'fef'

sql = "select instID, close from fut_daily where date like '2017-10-20%' and instID like 'fef____'"
df = pd.read_sql_query(sql, conn)  
df['instID'] = df['instID'].apply(lambda x: x[3:], 1)      
conn.close()           


"""
Summary:
    Given a day's future curve. Price all the spreads and averages
"""

ref = {'Q1':('01', '03'),                           # definitions of quaters and half-years
           'Q2':('04', '06'),
           'Q3':('07', '09'),
           'Q4':('10', '12'),
           'H1':('01', '06'),
           'H2':('07', '12'),
           'CAL':('01', '12')
           }

def foo(df, symbol):
    '''
    XXQ1
    XXH1
    XXCal
    XXXX, eg: 1801
    XXXX-XXXX eg: 1801-1804 
    '''
    symbol = str(symbol)
#    
    if '-' in symbol:                                  
        frm, to = symbol.split('-')
        df = df[(df['instID']>=frm)&(df['instID']<=to)]
        return df['close'].mean()
    
    year = symbol[0:2]
    tp = symbol[2:]
    try:                                              
        frm = year+ref[tp][0]
        to = year+ref[tp][1]        
        df = df[(df['instID']>=frm)&(df['instID']<=to)]
        return df['close'].mean()
    except KeyError:                                   
        return float(df[df['instID']==symbol]['close'].values)

def cal_spread(df, spread_pair):
    '''
    eg 18Q1/18Q2, 1801/18Q1
    '''
    minuend, subtraend = spread_pair.split('/')
    return foo(df, minuend.upper()) - foo(df, subtraend.upper())
    



