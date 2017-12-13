# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 09:26:57 2017

@author: junbai 

Analysis on stock '600837'
"""

import sys
sys.path.append("C:/Users/j291414/my algorithms")

import backtrader as bt
from datetime import datetime
from collections import OrderedDict
import pandas as pd
from backtrader.feeds import PandasData
import technical_analysis.technical_analysis as ta
import sqlite3
import matplotlib.pyplot as plt
from base import start_backtest
import strategies as st
import analyzers as ay

conn = sqlite3.connect("C:/Users/j291414/my algorithms/quant_trading/cn_stock.db")
code = '601988'
sql = "select Adj_Open as Open, Adj_Close as Close, Adj_High as High, Adj_Low as Low, Date \
        from cn_stocks_daily where code = '{}'".format(code)
df = pd.read_sql_query(sql, conn)
df['date'] = pd.to_datetime(df['Date'], format="%Y-%m-%dT%H:%M:%S")
df.index = df['date']
df = df[df.index>'2016-01-01']

start_backtest([df,], st.TestStrategy)
conn.close()
