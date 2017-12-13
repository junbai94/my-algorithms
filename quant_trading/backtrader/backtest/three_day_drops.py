# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:20:00 2017

@author: junbai

Three days drop strategy. Work towards to multiple assets
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

stock = get_stock_data('601988', frm='2017-01-01')
datafeed = bt.feeds.PandasData(dataname=stock)

start_backtest([datafeed,], strategy=st.TestStrategy, commission=0.004)