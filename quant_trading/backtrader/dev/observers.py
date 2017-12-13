# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:06:21 2017

@author: junbai

BackTrader Custom Observers
"""

import backtrader as bt

class OrderObserver(bt.observer.Observer):
    lines = ('created', 'expired',)
    
    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)
    
    plotlines = dict(
            created = dict(marker='*', markersize=8.0, color='lime', fillstyle='full'),
            expired = dict(marker='*', markersize=8.0, color='red', fillstyle='full')
            )
    
    def next(self):
        for order in self._owner._orderspending:
            if order.data is not self.data:
                continue
            if not order.isbuy():
                continue
            
            if order.status in [bt.Order.Accepted, bt.Order.Submitted]:
                self.lines.created[0] = order.created.price
                
            elif order.status in [bt.Order.Expired]:
                self.lines.expired[0] = order.created.price
                
                
class DummyObserver(bt.Observer):
    
    lines = ('counter',)
    
    params = (('haha', 1),)
    
    def next(self):
        self.p.haha += 1
        

class TotalPnLObserver(bt.Observer):
    
    lines = ('dummy', )
    
    plotinfo = dict(plot=True, subplot=True, plotlinelabels=True)
    
    def prenext(self):
        if not self.lines.dummy[0]:
           self.lines.dummy[0] = 0 
        
    def next(self):
        for trade in self._owner._tradespending:
            if trade.data not in self.datas:
                self.lines.dummy[0] = self.lines.dummy[-1]
            
            elif not trade.isclosed:
                self.lines.dummy[0] = self.lines.dummy[-1]
                
            else:
                self.lines.dummy[0] = trade.pnl + self.lines.dummy[-1]
                
            
            
            
    