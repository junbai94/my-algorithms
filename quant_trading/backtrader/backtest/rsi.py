# -*- coding: utf-8 -*-
"""
Created on Thu Dec 07 11:42:13 2017

@author: junbai

RSI signal on Chinese stocks
"""

import sys
sys.path.append("C:/Users/j291414/my algorithms/quant_trading/backtrader/dev")
sys.path.append("C:/Users/j291414/my algorithms/quant_trading/scripts")

import pandas as pd
import numpy as np
import backtrader as bt
from dbaccess import get_stock_data, get_stock_codes
from base import start_backtest
import strategies as st

stock = get_stock_data('601988', frm='2016-05-01')
stock.index = stock['Date']
datafeed = bt.feeds.PandasData(dataname=stock)

start_backtest([datafeed], params={'low':35.0}, commission=0.004, strategy=st.RSIStrategy)