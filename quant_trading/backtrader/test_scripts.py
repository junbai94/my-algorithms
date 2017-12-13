# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:59:45 2017

@author: junbai

Test Pair Trading
"""
import sys
sys.path.append("C:/Users/j291414/my algorithms")

import pandas as pd
from base import start_backtest
import strategies as st
import backtrader as bt
from data_handling.new_data import get_data, merge_data
from quant_trading.backtrader.dev import base


plt = get_data('plt_io62', 'spot_daily')
plt.index = plt['date']
tsi = get_data('tsi_io62', 'spot_daily')
tsi.index = tsi['date']
#merged = merge_data([plt,tsi])



start_backtest([plt, tsi], st.MultiDataTestStrategy, analysis=False)