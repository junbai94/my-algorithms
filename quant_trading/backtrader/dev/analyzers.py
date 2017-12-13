# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:56:51 2017

@author: junbai

BackTrader Analyzers visualiztion
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import itertools
import operator

import six
from six.moves import map, xrange, zip

import backtrader as bt
import backtrader.indicators as btind
from backtrader.utils import OrderedDict
import indicators as ind

def printTradeAnalysis(analyzer):
    '''
    Function to print the Technical Analysis results in a nice format.
    '''
    #Get the results we are interested in
    total_open = analyzer.total.open
    total_closed = analyzer.total.closed
    total_won = analyzer.won.total
    total_lost = analyzer.lost.total
    win_streak = analyzer.streak.won.longest
    lose_streak = analyzer.streak.lost.longest
    pnl_net = round(analyzer.pnl.net.total,2)
    strike_rate = (total_won / total_closed) * 100
    #Designate the rows
    h1 = ['Total Open', 'Total Closed', 'Total Won', 'Total Lost']
    h2 = ['Strike Rate','Win Streak', 'Losing Streak', 'PnL Net']
    r1 = [total_open, total_closed,total_won,total_lost]
    r2 = [strike_rate, win_streak, lose_streak, pnl_net]
    #Check which set of headers is the longest.
    if len(h1) > len(h2):
        header_length = len(h1)
    else:
        header_length = len(h2)
    #Print the rows
    print_list = [h1,r1,h2,r2]
    row_format ="{:<15}" * (header_length + 1)
    print
    print("Trade Analysis Results:")
    for row in print_list:
        print(row_format.format('',*row))

def printSQN(analyzer):
    sqn = round(analyzer.sqn,2)
    print
    print('SQN: {}'.format(sqn))
    
    
class LegDownUpAnalyzer(bt.Analyzer):
    params = (
        # If created indicators have to be plotteda along the data
        ('plotind', True),
        # period to consider for a legdown
        ('ldown', 10),
        # periods for the following legups after a legdown
        ('lups', [5, 10, 15, 20]),
        # How to sort: date-asc, date-desc, legdown-asc, legdown-desc
        ('sort', 'legdown-desc'),
    )

    sort_options = ['date-asc', 'date-des', 'legdown-desc', 'legdown-asc']

    def __init__(self):
        # Create the legdown indicator
        self.ldown = ind.LegDown(self.data, period=self.p.ldown)
        self.ldown.plotinfo.plot = self.p.plotind

        # Create the legup indicators indicator - writeback is not touched
        # so the values will be written back the selected period and therefore
        # be aligned with the end of the legdown
        self.lups = list()
        for lup in self.p.lups:
            legup = ind.LegUp(self.data, period=lup)
            legup.plotinfo.plot = self.p.plotind
            self.lups.append(legup)

    def nextstart(self):
        self.start = len(self.data) - 1

    def stop(self):
        # Calculate start and ending points with values
        start = self.start
        end = len(self.data)
        size = end - start

        # Prepare dates (key in the returned dictionary)
        dtnumslice = self.strategy.data.datetime.getzero(start, size)
        dtslice = map(lambda x: bt.num2date(x).date(), dtnumslice)
        keys = dtslice

        # Prepare the values, a list for each key item
        # leg down
        ldown = self.ldown.legdown.getzero(start, size)
        # as many legs up as requested
        lups = [up.legup.getzero(start, size) for up in self.lups]

        # put legs down/up together and interleave (zip)
        vals = [ldown] + lups
        zvals = zip(*vals)

        # Prepare sorting options
        if self.p.sort == 'date-asc':
            reverse, item = False, 0
        elif self.p.sort == 'date-desc':
            reverse, item = True, 0
        elif self.p.sort == 'legdown-asc':
            reverse, item = False, 1
        elif self.p.sort == 'legdown-desc':
            reverse, item = True, 1
        else:
            # Default ordering - date-asc
            reverse, item = False, 0

        # Prepare a sorted array of 2-tuples
        keyvals_sorted = sorted(zip(keys, zvals),
                                reverse=reverse,
                                key=operator.itemgetter(item))

        # Use it to build an ordereddict
        self.ret = OrderedDict(keyvals_sorted)

    def get_analysis(self):
        return self.ret
    
    def print(self, *args, **kwargs):
        # Overriden to change default behavior (call pprint)
        # provides a CSV printout of the legs down/up
        header_items = ['Date', 'LegDown']
        header_items.extend(['LegUp_%d' % x for x in self.p.lups])
        header_txt = ','.join(header_items)
        print(header_txt)

        for key, vals in six.iteritems(self.ret):
            keytxt = key.strftime('%Y-%m-%d')
            txt = ','.join(itertools.chain([keytxt], map(str, vals)))
            print(txt)
            
            
class PivotPointAnalyzer(bt.Analyzer):
    '''
    Analyzer to return some basic statistics showing:
        - Total days/periods analyzed
        - Percentage p was touched
        - Percentage s1 was touched
        - Percentage s2 was touched
        - Percentage r1 was touched
        - Percentage r2 was touched
        - Percentage s1 and r1 were touched on the same day / period
        - Percentage s1 and r2 were touched on the same day / period
        - Percentage s2 and r1 were touched on the same day / period
        - Percentage s2 and r1 were touched on the same day / period
        - Percentage p was touched without touching s1
        - Percentage p was touched without touching r1
        - Percentage s1 was touched without touching r1
        - Percentage r1 was touched without touching s1
    '''
    
    