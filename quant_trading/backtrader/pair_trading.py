# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 14:53:48 2017

@author: Junbai

Pair Trading/Mean Reverting based on z-score
Observer: cumulated PNL for every trade
Analyzer: Trade summary, Sharpe Ratio 
"""
import sys
sys.path.append("C:/Users/j291414/my algorithms/data_handling")
sys.path.append("C:/Users/j291414/my algorithms/quant_trading/backtrader/dev")

import pandas as pd
import backtrader as bt
from base import start_backtest
import strategies as st
import backtrader.indicators as btind
from backtrader.feeds import PandasData
import statsmodels.api as sm
import new_data as nd

class myPandas(PandasData):
    lines = ('spread',)
    params = (
        ('nocase', True),
        ('datetime', None),
        ('open', -1),
        ('high', -1),
        ('low', -1),
        ('close', -1),
        ('volume', -1),
        ('openinterest', -1),
        ('spread', -1),
    )        
    datafields = [
        'datetime', 'open', 'high', 'low', 'close', 'volume', 'openinterest',\
        'spread',
    ]

# get EU and CN index data
tianjin = nd.get_data('tj_hrc', 'spot_daily', 'tj', frm='2016-01-01')
shanghai = nd.get_data('sh_hrc', 'spot_daily', 'sh', frm='2016-01-01')
merged = nd.merge_data([tianjin, shanghai])
merged['spread'] = merged.tj - merged.sh
merged.index = merged.date

tj = pd.DataFrame({'close':merged.tj, 'spread':merged.spread})
sh = pd.DataFrame({'close':merged.sh, 'spread':merged['spread']})

data0 = myPandas(dataname=tj)
data1 = myPandas(dataname=sh)

if __name__ == '__main__':
    start_backtest([data0, data1], strategy=st.MultiDataTestStrategy, analysis=True)

