# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:38:15 2017

@author: junbai

BackTrader Strategies
"""

import sys
sys.path.append("C:/Users/j291414/my algorithms")

import pandas as pd
import backtrader as bt
import backtrader.feeds as btfeeds
import backtrader.indicators as btind
import numpy as np
import matplotlib.pyplot as plt
import indicators as ind

class BaseStrategy(bt.Strategy):
    
    def log(self, txt, dt=None):
        """ logging data """
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))
        
    def notify_order(self, order):        
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.3f, Cost: %.3f, Comm %.3f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.3f, Cost: %.3f, Comm %.3f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None
    
    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.3f, VALUE, %.3f' %
                 (trade.pnl, trade.pnlcomm, self.broker.getvalue()))  
        
    def start(self):
        print('Backtesting is about to start')

    def stop(self):
        print('Backtesting is finished')
        
    def log_order(self, isbuy=True):
        if isbuy:
            self.log("BUY CREATED, %.3f" % self.close[0])
        else:
            self.log("SELL CREATED, %.3f" % self.close[0])
    
    def order_at_close(self, isbuy=True):
        self.log_order(isbuy)
        if isbuy:
            order = self.buy(exectype=bt.Order.Close)
        else:
            order = self.sell(exectype=bt.Order.Close)
        return order
        
    def __init__(self):
        # keep reference of prices
        self.close = self.data.close
        
        # keep reference of order
        self.order = None
        self.buyprice = None
        self.sellprice = None
    
    def next(self):
        self.log('Close, %.3f' % self.close[0])


##############################################################################    
class TestStrategy(BaseStrategy):

    def notify_order(self, order):
        super(TestStrategy, self).notify_order(order)
        self.cost = order.executed.value

    def next(self):
        super(TestStrategy, self).next()

        if self.order:
            return

        if not self.position:
            if self.close[0] <= self.close[-1]:
                if self.close[-1] <= self.close[-2]:
                    if self.close[-2] <= self.close[-3]:
                        self.log('BUY CREATED')
                        self.order = self.buy()

        else:
            # cash out if rises
            if self.close[0] > self.close[-1]:
                self.log('SELL CREATED')
                self.order = self.sell()


###############################################################################               
class MovingAverageCrossover(BaseStrategy):
    params = (
            ('fast', 5),
            ('slow', 20),
            )
    
    def __init__(self):
        super(MovingAverageCrossover, self).__init__()
        self.sma_fast = bt.indicators.SimpleMovingAverage(self.close, period=self.params.fast)
        self.sma_slow = bt.indicators.SimpleMovingAverage(self.close, period=self.params.slow)
        
    def next(self):
        super(MovingAverageCrossover, self).next()
        
        # crossover strategy
        if self.order:
            return 
        
        if not self.position:
            if self.sma_fast < self.sma_slow:
                self.order = self.order_at_close()
                
        else:
            if self.sma_fast >= self.sma_slow:
                self.order = self.order_at_close(False)
                
                
###############################################################################
class MultiDataTestStrategy(BaseStrategy):
    def __init__(self):
        # keep reference of order
        self.order = None
        self.buyprice = None
        self.sellprice = None
        
        self.zscore = ind.Zscore(period=10)
        
    def next(self):
        self.log("CLOSE0, %.2f, CLOSE1, %.2f, ZSCORE, %.2f" % (self.data0.close[0], self.data1.close[0], self.zscore[0]))
  

        # test strategy
        if self.order:
            return 
        
        if not self.position:
            if self.zscore > 2.0:
                self.buy(data=self.data1, size=111, exectype=bt.Order.Close)
                self.sell(data=self.data0, size=100, exectype=bt.Order.Close)
                
            elif self.zscore < -2.0:
                self.buy(data=self.data0, size=100, exectype=bt.Order.Close)
                self.sell(data=self.data1, size=111, exectype=bt.Order.Close)
                
        else:
            if self.zscore<=0.5 and self.zscore >= -0.5:
                self.close(data=self.data0, exectype=bt.Order.Close)
                self.close(data=self.data1, exectype=bt.Order.Close)

###############################################################################
class TrixSignalStrategy(BaseStrategy):
    def __init__(self):
        super(TrixSignalStrategy, self).__init__()
        self.trix_9 = ind.MyTrixSignal(sigperiod=9)
        self.trix_15 = ind.MyTrixSignal(sigperiod=15)
        
    def next(self):
        self.log("CLOSE, %.2f, TRIX9, %.5f, TRIX15, %.5f" % (self.close[0], self.trix_9[0], self.trix_15[0]))
        
#        # Trix crossover
#        if self.order:
#            return 
#        
#        if not self.position:
#            if self.trix_9 < self.trix_15:
#                self.buy()
#        
#        else:
#            if self.trix_9 >= self.trix_15:
#                self.sell()
#    
                
###############################################################################
class RSIStrategy(BaseStrategy):
    
    params = dict(
            low = 30.0,
            high = 70.0
            )
    
    def __init__(self): 
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=21)
        # keep reference of order
        self.order = None
        self.buyprice = None
        self.sellprice = None
        
    def next(self):
        self.log("CLOSE, %.2f, RSI, %.2f" % (self.data.close[0], self.rsi[0]))
        
        # RSI strategy
        if self.order:
            return 
        
        if not self.position:
            if self.rsi < self.p.low:
                self.log("BUY CREATED")
                self.order = self.order_target_percent(target=0.9)

        else:
            if self.rsi > self.p.high:
                self.log("SELL CREATED")
                self.order = self.order_target_percent(target=0.0)
                
                
###############################################################################

    
            