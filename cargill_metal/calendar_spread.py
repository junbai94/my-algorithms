r# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:59:52 2017

@author: Junbai

This script is for computing calendar spread for futures
price are assumed following normal return model
"""
import pandas as pd
import sqlite3
from datetime import datetime, timedelta

DATABASE = 'C:\\Users\j291414\\Desktop\\market_data.db'

def initialize(symbol, contract, db):
    res = pd.DataFrame()
    conn = sqlite3.connect(db)
    for inst in contract:
        sql = "SELECT date, close AS {} \
               FROM fut_daily \
               WHERE instID = '{}'".format(symbol+str(inst),symbol+str(inst))
        df = pd.read_sql_query(sql, conn)
        if len(res) == 0:
            res = df
        else:
            res = res.merge(df, on='date')
        res.index = res['date']
    return res

def spread_engine(symbol, contracts, data):
    res = dict()
    for i in range(len(contracts)-1):
        for j in range(i+1, len(contracts)):
            res[symbol+str(contracts[i])+'/'+symbol+str(contracts[j])] = \
            data[symbol+str(contracts[i])] - data[symbol+str(contracts[j])]
    return res

def cutoff_date(contract):
    year = 2000 + contract/100
    month = contract % 100
    dt = (datetime(year, month, 1) + timedelta(days=-15)).strftime('%Y-%m-%d %H:%M:%S')
    return dt

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
def spread_table(tick_symbol = 'fef', contracts = [1701, 1702, 1703, 1704, 1705],
                 db = DATABASE):
    data = initialize(tick_symbol, contracts, db)   # DataFrame that records all future prices
    
    spreads = spread_engine(tick_symbol, contracts, data)   # dictionary that records all pairs' spread
    
    table = dict()
    for i in range(len(contracts)-1):
        for j in range(i+1, len(contracts)):
            spread = spreads[tick_symbol+str(contracts[i])+'/'+tick_symbol+str(contracts[j])]
            spread = spread.loc[spread.index <= cutoff_date(contracts[i])]
            spread_diff = spread.diff()
            table[tick_symbol+str(contracts[i])+'/'+tick_symbol+str(contracts[j])] = \
                [spread_diff[-20:].std(), spread_diff[-40:].std(), spread_diff[-60:].std()]
    
    res = pd.DataFrame(table)
    res.index = ['1M', '2M', '3M']
    return res
    
if __name__ == '__main__':
    pass
#    res = spread_table()