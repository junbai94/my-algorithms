# -*- coding: utf-8 -*-
"""
Created on Wed Nov 15 16:58:37 2017

@author: junbai

Alert Mean Reverting Trading Signal for Pair Trading
"""
import sys
sys.path.append("C:/Users/j291414/my algorithms")

import statsmodel.formula.api as sm
import pandas as pd
import numpy as np
from data_handling.new_regression import *

def mean_reverting_alert(df):
    """
    Pre-condition:
        df - pandas DataFrame with first column as data, second column as dependent
        variable. Third as independent variable
    """
    reg = Regression(df)
    k = reg.result.params[1]
    b = reg.result.params[0]
    

