# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 09:26:21 2017

@author: Junbai

This script performs regression analysis on two Data object
"""
import statsmodels.formula.api as sm
from dateutil import parser
import matplotlib.pyplot as plt

class Regression(object):
    def __init__(self, dep, indep):
        self.dep = dep
        self.indep = indep
        self.compare = dep.merge(indep, on='date', suffixes = ('_dep', '_indep'))
        self.result = sm.ols("price_dep ~ price_indep", self.compare).fit()
                
        
    def params(self):
        return self.result.params
    
    def summary(self):
        return self.result.summary()
    
    def scatter_plot(self):
        self.compare.plot(x='price_indep', y='price_dep', kind='scatter')
        
    def coef(self):
        return self.result.params['price_indep']
    
    def intercept(self):
        try:
            return self.result.params['Intercept']
        except KeyError:
            return 0
    
    def error_stats(self):
        coef = self.coef()
        intercept = self.intercept()
        error = self.compare['price_dep'] - coef*self.compare['price_indep']- intercept
        error.index = self.compare['date']
        return error.describe()
    
    def error_plot(self):
        coef = self.coef()
        intercept = self.intercept()
        error = self.compare['price_dep'] - coef*self.compare['price_indep']- intercept
        error.index = self.compare['date']
        error.plot()
        
    def date_range(self, start, end):
        start = parser.parse(start)
        end = parser.parse(end)
        temp = self.dep.df
        temp = temp.loc[(temp['date']>=start)&(temp['date']<=end)]
        self.dep.df = temp
        
        temp = self.indep.df
        temp = temp.loc[(temp['date']>=start)&(temp['date']<=end)]
        self.indep.df = temp
        
    def reduce_dof(self):
        self.result = sm.ols("price_dep ~ price_indep -1", self.compare).fit()