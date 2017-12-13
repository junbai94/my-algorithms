# -*- coding: utf-8 -*-
"""
Created on Tue Nov 28 11:49:50 2017

@author: junbai

EU and CN HRC Index Pair 
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

# Reference
# https://estrategiastrading.com/oro-bolsa-estadistica-con-python/

import datetime
import pandas as pd

import scipy.stats

import backtrader as bt


class PearsonR(bt.ind.PeriodN):
    _mindatas = 2  # hint to the platform

    lines = ('correlation',)
    params = (('period', 20),)

    def next(self):
        c, p = scipy.stats.pearsonr(self.data0.get(size=self.p.period),
                                    self.data1.get(size=self.p.period))

        self.lines.correlation[0] = c


class MACrossOver(bt.Strategy):
    params = (
        ('ma', bt.ind.MovAv.SMA),
        ('pd1', 20),
        ('pd2', 20),
    )

    def __init__(self):
        ma1 = self.p.ma(self.data0, period=self.p.pd1, subplot=True)
        self.p.ma(self.data1, period=self.p.pd2, plotmaster=ma1)
        PearsonR(self.data0, self.data1)


def runstrat(args=None):

    cerebro = bt.Cerebro()

    df = pd.read_csv("C:/Users/j291414/Desktop/data.csv")
    df.index = pd.to_datetime(df['Date'], format="%d/%m/%Y")
    df0 = pd.DataFrame({'close':df['EU']})
    df1 = pd.DataFrame({'close':df['CN']})
    data0 = bt.feeds.PandasData(dataname=df0)
    data1 = bt.feeds.PandasData(dataname=df1)
    data1.plotinfo.plotmaster = data0
    cerebro.adddata(data0)
    cerebro.adddata(data1)



    # Strategy
    cerebro.addstrategy(MACrossOver)

    cerebro.addobserver(bt.observers.LogReturns2,
                        timeframe=bt.TimeFrame.Weeks,
                        compression=20)

    # Execute
    cerebro.run()

    cerebro.plot()
    
runstrat()