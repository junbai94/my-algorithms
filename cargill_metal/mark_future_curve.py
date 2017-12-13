# -*- coding: utf-8 -*-
"""
Created on Tue Oct 31 09:41:37 2017

@author: junbai

Mark future curve -  test
"""

import pandas as pd
import numpy as np
from numpy.linalg import inv
from datetime import datetime

#MONTH_TENORS = ['1710', '1711', '1712', '1801', '1802', '1803', '1804', '1805', 
#                '1806', '1807', '1808', '1809', '1810', '1811', '1812']
#
#SPREAD_TENORS = ['1801', '1710/1711', '1711/1712', '1712/1801', '1801/1802',
#         '1802/1803', '1803/1804', '18Q1/18Q2', '18Q2/18Q3', '18Q3/18Q4',
#         '1804/1805/1806', '1806/1807/1808', '1807/1808/1809', 
#         '1809/1810/1811', '1810/1811/1812']
#
#SPREADS = (461.0, 0.55, 0.4, 0.4, 0.4, 0.4, 0.4, 1.2, 
#               1.05, 0.9, 0.05, 0.05, 0.0, 0.0, 0.0)

"""
Summary:
   Call function *get_curve* to mark future curve. To convert a list of date ojects
   to standard list of month tenors, call function *to_month_tenors*
   
Pre-conditions:
    :month_tenors are a list of string of the months you want to compute.
    :spread_pair and spread value should be one to one corresponsed.
    :Input spread_pairs should ensure the desired output is computable.
    
Post-conditions:
    :output of get_curve function is a DataFrame object with the column of prices
        named 'close'
    :output can be fed back to functions in script - compute_calendar_spread to 
        calculate spread
"""        

REF = {
       'Q1': ['01', '02', '03'],
       'Q2': ['04', '05', '06'],
       'Q3': ['07', '08', '09'],
       'Q4': ['10', '11', '12'],
       'H1': ['01', '02', '03', '04', '05', '06'],
       'H2': ['07', '08', '09', '10', '11', '12'],
       'CAL': ['01', '02', '03', '04', '05', '06',
               '07', '08', '09', '10', '11', '12']
       }

def pos(month_tenors, tenor):
    return month_tenors.index(tenor)

def month_row(month_tenors, tenor):
    size = len(month_tenors)
    mat = np.matlib.zeros(size)
     
    # self defined range like '1710-1802'           
    if '-' in tenor:
        frm, to = tenor.split('-')
        frm_mth = int(frm[2:])
        to_mth = int(to[2:])
        frm_yr = int(frm[:2])
        to_yr = int(to[:2])
        
        if to_yr > frm_yr:
            length = 12 + to_mth - frm_mth + 1
        else:
            length = to_mth - frm_mth + 1
        
        pos_start = pos(month_tenors, frm)
        position = np.arange(pos_start, pos_start+length).tolist()
        for e in position:
            mat[0, e] = 1.0/length
    
    # eg Q1 or H1 or CAL            
    else:
        year = tenor[:2]
        label = tenor[2:]
        if label in REF.keys():
            length = len(REF[label])
            position = [pos(month_tenors, year+x) for x in REF[label]]
            for e in position:
                mat[0, e] = 1.0/length
        
        # eg normal month like '1701' or '1812'       
        else:
            position = pos(month_tenors, tenor)
            mat[0, position] = 1.0
    return mat
    
def spread_row(month_tenors, spread):
    splitted = spread.split('/')
    if len(splitted) == 1:
        return month_row(month_tenors, splitted[0])
    elif len(splitted) == 2:
        return month_row(month_tenors, splitted[0]) - month_row(month_tenors, splitted[1])
    elif len(splitted) == 3:
        return month_row(month_tenors, splitted[0]) - 2* month_row(month_tenors, splitted[1]) + month_row(month_tenors, splitted[2])
    else:
        raise ValueError("INPUT TENOR IS NOT CORRECT")   
        
def generate_matrix(month_tenors, spread_pairs):
    res = None
    for tenor in spread_pairs:
        if res is None:
            res = spread_row(month_tenors, tenor)
        else:
            res = np.vstack((res, spread_row(month_tenors, tenor)))
    return res

def get_curve(month_tenors, spread_pairs, spreads):
    sprd = np.matrix(spreads).transpose()
    matrix = generate_matrix(month_tenors, spread_pairs)
    curve = np.dot(inv(matrix), sprd).transpose().tolist()[0]
    return pd.DataFrame({'instID':month_tenors,
                         'close':curve})[['instID', 'close']]
    