# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:38:15 2017

@author: junbai

plot custom-created indicators
"""
import sys
sys.path.append("../../dev")

import datetime
import pandas as pd
from base import start_backtest
import strategies as st
import indicators as ind
import observers as ob
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
import backtrader as bt

df = pd.read_csv("../data/AAPL.csv")
df.index = pd.to_datetime(df['Date'], format="%d/%m/%Y")

class MyStrategy(st.BaseStrategy):
    params = dict(
                smaperiod = 15,
                limitperc = 1.0,
                valid = 7,
                )
    
    def __init__(self):
        super(MyStrategy, self).__init__()
        sma = btind.SMA(period=self.p.smaperiod)
        self.buysell = btind.CrossOver(self.data.close, sma, plot=True)
        
    def next(self):
        super(MyStrategy, self).next()
        
        # Check if we are in the market
        if self.position:
            if self.buysell < 0:
                self.log('SELL CREATE, %.2f' % self.data.close[0])
                self.sell()

        elif self.buysell > 0:
            plimit = self.data.close[0] * (1.0 - self.p.limitperc / 100.0)
            valid = self.data.datetime.date(0) + \
                datetime.timedelta(days=self.p.valid)
            self.log('BUY CREATE, %.2f' % plimit)
            self.buy(exectype=bt.Order.Limit, price=plimit, valid=valid)
        
cerebro = bt.Cerebro()

data = bt.feeds.PandasData(dataname=df)
cerebro.adddata(data)

cerebro.addobserver(ob.OrderObserver)

cerebro.addstrategy(MyStrategy)
cerebro.run()

cerebro.plot()