# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 17:13:57 2017

@author: junbai

New Data computation class
"""
import sys
sys.path.append("C:/Users/j291414/my algorithms")

import datetime
import pandas as pd
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from risk_engine import misc
import new_regression as nr
from statsmodels.tsa.seasonal import seasonal_decompose
import warnings
pd.options.mode.chained_assignment = None  # default='warn'
warnings.filterwarnings("ignore", category=DeprecationWarning) 
warnings.filterwarnings("ignore", category=FutureWarning) 


DATABASE = "C:/Users/j291414/Desktop/market_data.db"

def get_data(ID, table='spot_daily', name=None, frm=None, to=None, to_USD=False, rolling=None, db=DATABASE):
    conn = sqlite3.connect(db)
    if table == 'fx_daily':
        sql = "select date, rate from fx_daily where tenor='0W'"
        temp = pd.read_sql_query(sql, conn)
    elif table == 'spot_daily':
        sql = "select date, close from spot_daily where spotID = '{}'".format(ID)
        temp = pd.read_sql_query(sql, conn)
    elif table == 'fut_daily':
	if not 'hc' in ID: 
            sql = "select date, close from fut_daily where instID = '{}'".format(ID)
	else:
	    sql = "select date, close from fut_daily where instID = '{}' and exch='SHFE'".format(ID)
        temp = pd.read_sql_query(sql, conn)
    elif table == 'spot_index':
        sql = "select date, close from spot_index where code = '{}'".format(ID)
        temp = pd.read_sql_query(sql, conn)
    else:
        raise ValueError('FUNCTION NOT SUPPORT THIS TABLE')
    
    temp['date'] = pd.to_datetime(temp['date'], format='%Y-%m-%d %H:%M:%S')
    
    if to_USD:
        fx = get_data('fx', 'fx_daily')
        temp = divide_fx(temp, fx)
        temp = temp[['date', 'result']]

    if rolling:
	temp.iloc[:,1] = pd.rolling_mean(temp.iloc[:,1], window=rolling)

    if name:
        temp.columns = ['date', name]

    if frm or to:
	temp = date_range(temp, frm, to)

    conn.close()
    return temp


def get_cont_contract(ticker, n, frm, to, rolling_rule='-30b', \
                      freq='d', need_shift=False, name=None, to_USD=False):
    frm = datetime.strptime(frm, "%Y-%m-%d").date()
    to = datetime.strptime(to, "%Y-%m-%d").date()
    temp = misc.nearby(ticker, n, frm, to, rolling_rule, freq, need_shift)
    temp['date'] = pd.to_datetime(temp.index, format='%Y-%m-%d %H:%M:%S')
    temp = temp[['date', 'close', 'contract']]
    
    if to_USD:
        fx = get_data('fx', 'fx_daily')
        temp = temp.merge(fx, on='date')
        temp['result'] = temp['close'].divide(temp['rate'])
        temp = temp[['date', 'result', 'contract']]
        
    if name:
       temp.columns = ['date', name, 'contract']
    
    temp.index = range(len(temp))  
    return temp


def merge_data(df_list):
    temp = df_list[0]
    for i in range(1, len(df_list)):
        temp = temp.merge(df_list[i], on='date')
    temp = temp.dropna()
    return temp

def date_range(df, frm=None, to=None):
    temp = df
    try:
        if frm:
	    temp = temp[temp['date'] >= frm]
	if to:
	    temp = temp[temp['date'] <= to]
        return temp

    except KeyError:
        raise KeyError("NAME YOUR DATE COLUMN TO BE 'date'")  
        
def monthly_avg(df):
    return df.resample("M", how ='mean', on='date')

def times_fx(df, fx):
    temp = df.merge(fx, on='date')
    if 'result' not in temp.columns:
        temp['result'] = temp.iloc[:,1].multiply(temp.iloc[:,2])
        return temp
    else:
        raise ValueError('Do not choose result as column name')
        
def divide_fx(df, fx):
    temp = df.merge(fx, on='date')
    if 'result' not in temp.columns:
        temp['result'] = temp.iloc[:,1].divide(temp.iloc[:,2])
        return temp
    else:
        raise ValueError('Do not choose result as column name')
        
def plot_data(df_list, enlarge=True):
    if enlarge:
        plt.figure(figsize=(10, 5))
    for df in df_list:
        label = df.columns[1]
        plt.plot(df['date'], df[label], label=label)
    plt.xticks(rotation='vertical')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def quick_regression(dep_tpl, indep_tpl):
    if type(dep_tpl) == type(tuple()):
        dep = get_data(dep_tpl[0], dep_tpl[1], name=dep_tpl[2])
        indep = get_data(indep_tpl[0], indep_tpl[1], indep_tpl[2])
	name_dep = dep_tpl[2]
	name_indep = indep_tpl[2]
    else:
	dep = dep_tpl
	indep = indep_tpl
	name_dep = dep.columns[1]
	name_indep = indep.columns[1]
    merged = merge_data([dep, indep])
    reg = nr.Regression(merged, name_dep, [name_indep,])
    reg.run_all()
    return reg

def quick_regression_analysis(dep_tpl, indep_tpl):
    if type(dep_tpl) == type(tuple()):
        dep = get_data(dep_tpl[0], dep_tpl[1], name=dep_tpl[2])
        indep = get_data(indep_tpl[0], indep_tpl[1], indep_tpl[2])
	name_dep = dep_tpl[2]
	name_indep = indep_tpl[2]
    else:
	dep = dep_tpl
	indep = indep_tpl
	name_dep = dep.columns[1]
	name_indep = indep.columns[1]
    merged = merge_data([dep, indep])
    reg = nr.Regression(merged, name_dep, [name_indep,])
    reg.summarize_all()
    return reg