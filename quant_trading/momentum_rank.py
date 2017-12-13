# -*- coding: utf-8 -*-
"""
Created on Mon Dec 04 16:14:08 2017

@author: junbai

Rank stocks based on slope
"""

import sys
sys.path.append("C:/Users/j291414/my algorithms/quant_trading/scripts")

import pandas as pd
import sqlite3
from dbaccess import get_stock_data, get_stock_codes
from momentum_slope_ranking import slope, main
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning)

codes = get_stock_codes()

result = main(list(codes['Code']))