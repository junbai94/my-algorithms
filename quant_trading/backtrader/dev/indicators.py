# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 11:21:18 2017

@author: junbai

BackTrader Indicators
"""

import backtrader as bt
import backtrader.indicators as btind
import numpy as np
import pandas as pd

class MyTrix(bt.Indicator):
    
    lines = ('trix',)
    params = (('period', 15), )
    
    def __init__(self):
        ema1 = btind.EMA(self.data, period=self.p.period)
        ema2 = btind.EMA(ema1, period=self.p.period)
        ema3 = btind.EMA(ema2, period=self.p.period)
        
        self.lines.trix = 100.0 * (ema3 - ema3(-1)) / ema3(-1)
        
        
        
class MyTrixSignal(MyTrix):
    
    lines = ('signal', )
    params = (('sigperiod', 9),)
    
    def __init__(self):
        print(self.p.sigperiod)
        super(MyTrixSignal, self).__init__()
        self.lines.signal = btind.EMA(self.lines.trix, period=self.p.sigperiod)


class LegDown(bt.Indicator):
    
    lines = ('legdown', )
    params = (('period', 10),)
    
    def __init__(self):
        self.lines.legdown = self.data.high(-self.p.period) - self.data.low
     
        
class LegUp(bt.Indicator):
    
    lines = ('legup', )
    params = (('period', 10), ('writeback', True), )
    
    def __init__(self):
        self.lu = self.data.high - self.data.low(-self.p.period)
        self.lines.legup = self.lu(self.p.period * self.p.writeback)

class Zscore(bt.Indicator):
    lines = ("zscore", )
    params = (('period', 10), )

    def __init__(self):
        spread = self.data0.spread
        spreadMean = bt.indicators.SMA(spread, period=self.p.period) 
        spreadStd = bt.indicators.StandardDeviation(spread, period=self.p.period)
        
        self.lines.zscore = (spread - spreadMean)/spreadStd
        
class Log(bt.Indicator):
    lines = ("logarithm",)
    
    def next(self):
        self.lines.logarithm[0] = np.log(self.datas[1].close[0])
        
        
#class OLS_Slope_InterceptN(bt.Indicator):
#    '''
#    Calculates a linear regression using ``statsmodel.OLS`` (Ordinary least
#    squares) of data1 on data0
#
#    Uses ``pandas`` and ``statsmodels``
#
#    Use ``prepend_constant`` to influence the paramter ``prepend`` of
#    sm.add_constant
#    '''
#    _mindatas = 2  # ensure at least 2 data feeds are passed
#
#    packages = (
#        ('pandas', 'pd'),
#        ('statsmodels.api', 'sm'),
#    )
#    lines = ('slope', 'intercept',)
#    params = (
#        ('period', 10),
#        ('prepend_constant', True),
#    )
#
#    def next(self):
#        p0 = pd.Series(self.data0.get(size=self.p.period))
#        p1 = pd.Series(self.data1.get(size=self.p.period))
#        p1 = sm.add_constant(p1, prepend=self.p.prepend_constant)
#        slope, intercept = sm.OLS(p0, p1).fit().params
#
#        self.lines.slope[0] = slope
#        self.lines.intercept[0] = intercept


class OLS_Slope_Intercept(bt.Indicator):
    _mindatas = 2
    lines = ('slope', 'intercept', )
    
    def next(self):
        pass
        