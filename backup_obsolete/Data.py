# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 11:39:24 2017

@author: j291414


+++++++++++++++++                MANUAL                +++++++++++++++++++++++

This class perform basic data analysis on pandas DataFrame
 
Available analysis tool (more implementations to be added gradually):
1) Select data from a desired date range
2) Monthly average 
3) N-business-days average
4) N-days average
5) update or restore data of an object of this class
6) calculate time spread (monthly average)
7) subtracted/multiplied by another Data object (inner merge)
    
Initialization methods:
1) Simply select all data of target product using SQL SELECT phrase. You can
further refine your selection range using provided analysis tools
2) You can initialize an empty class object and substantialize its DataFrame 
data variable later

+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""

import pandas as pd
import sys
from dateutil import parser
import datetime
import math

class Data(object):

    # constructor
    def __init__(self, sql=None, conn=None):
        if sql is not None:
            self.df = self.read(sql, conn)
            self.archive = self.df
        else:
            self.df = None
            self.archive = None
    
    def read(self, sql, conn):
        df = pd.read_sql_query(sql, conn)
        try:
            df = df.loc[:,['date', 'price']]
        except KeyError:
            print("please name the columns as 'date', 'price'")
            sys.exit(0)
        temp = df['date']
        df['date'] = pd.Series([parser.parse(x) for x in temp])
        return df
    
    # getters
    def restore(self):
        self.df = self.archive.copy(False)
    
    def average_monthly(self):
        return self.df.resample("M", how ='mean', on='date')
    
    def average_biz_day(self, n):
        """
        average price of past n trading days
        """
        d = 0
        num = len(self.df['date'])
        price = self.df['price']
        date = self.df['date']
        count, total = 0, 0
        temp_date, temp_price = [], []
        for d in range(num):
            count += 1
            total += price[d]
            if count == n:
                temp_price.append(total/count)
                temp_date.append(date[d])
                count, total = 0, 0
            if d+1==num and count!=0:
                temp_price.append(total/count)
                temp_date.append(date[d])
        return pd.Series(temp_price, temp_date)

    def average_period(self, n):
        temp_date = []
        temp_price = []
        df = self.df
        df = df.sort_values('date')
        start = df['date'].iloc[0]-datetime.timedelta(days=1)
        end = df['date'].iloc[0]+datetime.timedelta(days=n)
        while True:
            if start >= df['date'].iloc[-1]:
                break
            temp = df.loc[(df['date']>start)&(df['date']<=end)]
            if math.isnan(temp['price'].mean()):
                temp_price.append(temp_price[-1])
            else:
                temp_price.append(temp['price'].mean())
            temp_date.append(end)
            start = end
            end = end + datetime.timedelta(days=n)
        return pd.Series(temp_price, temp_date)
    
    def time_spread(self, n):
        mthly_avg = self.average_monthly()
        m = mthly_avg['price'][:-n]
        m_n = mthly_avg['price'][n:]
        sprd = []
        for a, b in zip(m, m_n):
            sprd.append(b-a)
        return pd.Series(sprd, mthly_avg.index[:-n])
   
    
    # setters
    def date_range(self, begin, end):
        """ 
        begin - start date string
        end - end date string
        """
        start = parser.parse(begin)
        end = parser.parse(end)
        temp = self.df
        temp = temp.loc[(temp['date']>=start)&(temp['date']<=end)]
        self.df = temp
       
    def set_df(self, df):
        try:
            df = df.loc[:,['date', 'price']]
        except KeyError:
            print("please name the columns as 'date', 'price'")
            sys.exit(0)     
        self.df = df
        self.archive = df
        
    def subtract(self, feed):
        temp = self.df.merge(feed.df, on='date')
        self.df = pd.DataFrame({'date':temp['date'],
                             'price':(temp['price_x']-temp['price_y'].values)})
    
    def times(self, feed):
        temp = self.df.merge(feed.df, on='date')
        self.df = pd.DataFrame({'date':temp['date'],
                             'price':(temp['price_x']*temp['price_y'].values)})
   

