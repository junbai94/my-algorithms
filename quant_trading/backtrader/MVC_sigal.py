# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 14:54:12 2017

@author: junbai

Moving average crossover signal
"""

import pandas as pd
import backtrader as bt

class CrossoverStrategy(bt.Strategy):
    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enougth cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None
    
    def log(self, txt, dt=None):
        dt = dt or self.data.datetime.date()
        print("%s, %s" % (dt, txt))

    def __init__(self):
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=21)

    def next(self):
        self.log('Close, %.2f, RSI, %.2f' % (self.data.close[0], self.rsi[0]))
        
        if not self.position:
            if self.rsi < 30:
                self.order = self.order_target_percent(target=1.0)

        else:
            if self.rsi > 70:
                self.order = self.order_target_percent(target=0.0)
                

if __name__ == '__main__':
    cerebro = bt.Cerebro()
    
    data = pd.read_csv("C:/Users/j291414/Desktop/sample_data.csv")
    data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")
    data = data[data['Date'] > '2017-01-01']
    data.index = data['Date']
    data = bt.feeds.PandasData(dataname=data)
    cerebro.adddata(data)
    
    cerebro.addstrategy(CrossoverStrategy)
    
    cerebro.broker.setcash(1000000.0)
    
    print ("initial wealth: %.2f" % cerebro.broker.getvalue())
    
    cerebro.run()
    
    print ("final wealth: %.2f" % cerebro.broker.getvalue())