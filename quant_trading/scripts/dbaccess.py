# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 14:44:29 2017

@author: junbai

Get stock data from database
"""

import pandas as pd
import sqlite3
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

DATABASE_PATH = "C:/Users/j291414/Desktop/cn_stock.db"

def get_stock_data(code, frm=None, to=None, index=True):
    conn = sqlite3.connect(DATABASE_PATH)
    sql = "select Date, Adj_Open as Open, Adj_High as High, Adj_Low as Low, Adj_Close as Close, \
            Turnover_Volume as Volume from cn_stocks_daily \
            where Code = '{}'".format(code)
    temp = pd.read_sql_query(sql, conn)
    temp['Date'] = pd.to_datetime(temp['Date'], format="%Y-%m-%dT%H:%M:%S")
    
    if frm:
        temp = temp[temp['Date'] >= frm]
    if to:
        temp = temp[temp['Date'] <= to]
        
    if index:
        temp.index = temp['Date']
    
    conn.close()
    return temp

def get_stock_codes():
    conn = sqlite3.connect(DATABASE_PATH)
    sql = "select distinct(Code) from cn_stocks_daily"
    
    df = pd.read_sql_query(sql, conn)
    conn.close()
    return df
