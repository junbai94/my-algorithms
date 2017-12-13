# -*- coding: utf-8 -*-
"""
Created on Wed Dec 06 14:55:51 2017

@author: junbai

Test multi stocks ranking strategy

Pick the best momentum stock out of three. Daily adjustment
"""

import sys
sys.path.append("C:/Users/j291414/my algorithms/data_handling")
sys.path.append("C:/Users/j291414/my algorithms/quant_trading/backtrader/dev")

import numpy as np
import pandas as pd
import backtrader as bt
from base import start_backtest
import strategies as st
import backtrader.indicators as btind
from backtrader.feeds import PandasData
import statsmodels.api as sm
import new_data as nd
import sqlite3
import datetime


# get three stocks
conn = sqlite3.connect("C:/Users/j291414/Desktop/cn_stock.db")
codes = ['600050', '600104', '600519']
sql = "select Date, Adj_Open as Open, Adj_Close as Close, Adj_High as High, Adj_Low as low \
        from cn_stocks_daily where Code = '{}' and Date >= '2017-01-01T00:00:00'"
first = pd.read_sql_query(sql.format(codes[0]), conn)
first.index = pd.to_datetime(first['Date'], format='%Y-%m-%dT%H:%M:%S')
second = pd.read_sql_query(sql.format(codes[1]), conn)
second.index = pd.to_datetime(second['Date'], format='%Y-%m-%dT%H:%M:%S')
third = pd.read_sql_query(sql.format(codes[2]), conn)
third.index = pd.to_datetime(third['Date'], format='%Y-%m-%dT%H:%M:%S')
conn.close()


# strategy
#class TestStrategy(st.BaseStrategy):
#    def __init__(self):
#        self.first = self.getdatabyname('600050')
#        self.second = self.getdatabyname(codes[1])
#        self.third = self.getdatabyname(codes[2])
#        
#        
#    def next(self):
#        first_ret = (self.first.close[0] - self.first.close[-1]) / self.first.close[-1]
#        second_ret = (self.second.close[0] - self.second.close[-1]) / self.second.close[-1]
#        third_ret = (self.third.close[0] - self.third.close[-1]) / self.third.close[-1]
#        self.log("FIRST, %.2f, SECOND, %.2f, THIRD, %.2f" % (first_ret, second_ret, third_ret))
#        
#        if first_ret >= second_ret and first_ret >= third_ret:
#            self.close(data=self.second)
#            self.close(data=self.third)
#            self.order_target_percent(data=self.first, target=0.8)
#        
#        if second_ret >= first_ret and second_ret >= third_ret:
#            self.close(data=self.first)
#            self.close(data=self.third)
#            self.order_target_percent(data=self.second, target=0.8)
#            
#        if third_ret >= first_ret and third_ret >= second_ret:
#            self.close(data=self.second)
#            self.close(data=self.first)
#            self.order_target_percent(data=self.third, target=0.8)

class TestSizer(bt.Sizer):
    params = dict(stake=1)
    
    def _getsizing(self, comminfo, cash, data, isbuy):
        dt, i = self.strategy.datetime.date(), data._id
        s = self.p.stake * (1 + (not isbuy))
        print ("{} Data {} OType {} Sizing to {}".format(dt, data._name, ('buy' * isbuy) or 'sell', s))
        return s
    
    
class St(bt.Strategy):
    params = dict(
            enter = [1, 3, 4],
            hold = [7, 10, 15],
            usebracket = True,
            rawbracket = True,
            pentry = 0.015,
            plimits = 0.03,
            valid = 10,
            )
    
    def notify_order(self, order):
        if order.status == order.Submitted:
            return 
        
        dt, dn = self.datetime.date(), order.data._name
        print('{} {} Order {} Status {}'.format(dt, dn, order.ref, order.getstatusname()))

        whichord = ['main', 'stop', 'limit', 'close']
        if not order.alive():
            dorders = self.o[order.data]
            idx = dorders.index(order)
            dorders[idx] = None
            print('-- No longer alive {} Ref'.format(whichord[idx]))
            
            if all(x is None for x in dorders):
                dorders[:] = []  # empty list - New orders allowed
    
    def __init__(self):
        self.o = dict()
        self.holding = dict()
        
    def next(self):
        for i, d in enumerate(self.datas):
            dt, dn = self.datetime.date(), d._name
            pos = self.getposition(d).size
            print ('{} {} Position {}'.format(dt, dn, pos))
            
            if not pos and not self.o.get(d, None):
                if dt.weekday() == self.p.enter[i]:
                    if not self.p.usebracket:
                        self.o[d] = [self.buy(data=d)]
                        print ('{} {} Buy {}'.format(dt, dn, self.o[d][0].ref))
                        
                    else:
                        p = d.close[0]
                        pstp = p * (1.0 - self.p.plimits)
                        plmt = p * (1.0 + self.p.plimits)
                        valid = datetime.timedelta(self.p.valid)
                        
                        if self.p.rawbracket:
                            o1 = self.buy(data=d, exectype=bt.Order.Limit, price=p, valid=valid, transmit=False)
                            o2 = self.sell(data=d, exectype=bt.Order.Stop,
                                           price=pstp, size=o1.size,
                                           transmit=False, parent=o1)

                            o3 = self.sell(data=d, exectype=bt.Order.Limit,
                                           price=plmt, size=o1.size,
                                           transmit=True, parent=o1)
                            self.o[d] = [o1, o2, o3]
                        else:
                            self.o[d] = self.buy_bracket(
                                data=d, price=p, stopprice=pstp,
                                limitprice=plmt, oargs=dict(valid=valid))
                        print('{} {} Main {} Stp {} Lmt {}'.format(
                            dt, dn, *(x.ref for x in self.o[d])))
                    self.holding[d] = 0
                    
            elif pos:  # exiting can also happen after a number of days
                self.holding[d] += 1
                if self.holding[d] >= self.p.hold[i]:
                    o = self.close(data=d)
                    self.o[d].append(o)  # manual order to list of orders
                    print('{} {} Manual Close {}'.format(dt, dn, o.ref))
                    if self.p.usebracket:
                        self.cancel(self.o[d][1])  # cancel stop side
                        print('{} {} Cancel {}'.format(dt, dn, self.o[d][1]))


# main function
def run_strat():
    cerebro = bt.Cerebro()
    
    data = bt.feeds.PandasData(dataname=first)
    cerebro.adddata(data, name='600050')
    data = bt.feeds.PandasData(dataname=second)
    cerebro.adddata(data, name='600104')
    data = bt.feeds.PandasData(dataname=third)
    cerebro.adddata(data, name='600519')
    
    
    cerebro.broker.setcommission(0.004)
    cerebro.broker.setcash(100000)
    
    cerebro.addstrategy(St)
    
    cerebro.addsizer(TestSizer)
    
    print ('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    cerebro.run()
    print ('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    
    
if __name__ == '__main__':
    run_strat()