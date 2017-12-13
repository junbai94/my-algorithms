# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 10:38:15 2017

@author: junbai

plot custom-created indicators
"""
import sys
sys.path.append("../../dev")

import pandas as pd
from base import start_backtest
import strategies as st
import indicators as ind

df = pd.read_csv("../data/AAPL.csv")
df.index = pd.to_datetime(df['Date'], format="%d/%m/%Y")

class NullStrategy(st.BaseStrategy):
    params = (
            ('trixperiod', 15),
            )
    
    def __init__(self):
        super(NullStrategy, self).__init__()
        ind.MyTrixSignalInherited(self.data, period=self.p.trixperiod)
        
if __name__ == '__main__':
    start_backtest([df,], NullStrategy, analysis=False)