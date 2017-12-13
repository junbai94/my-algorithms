# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 14:13:06 2017

@author: junbai

Regression Analysis
"""

import sqlite3
import pandas as pd
from statsmodels.formula.api import ols
import matplotlib.pyplot as plt
import numpy as np
import new_data
from pykalman import KalmanFilter
from scipy import poly1d
from stats_test import test_mean_reverting, half_life
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from statsmodels.tsa.stattools import coint, adfuller

DATABASE = "C:/Users/j291414/Desktop/market_data.db"

        
class Regression(object):
    
    def __init__(self, df, dependent=None, independent=None):
        """
        Initialize the class object
        Pre-condition:
            dependent - column name
            independent - list of column names
        """
        if not dependent:
            dependent = df.columns[1]
        if not independent:
            independent = [df.columns[2],]
            
        formula = '{} ~ '.format(dependent)
        first = True
        for element in independent:
            if first:
                formula += element
                first = False
            else:
                formula += ' + {}'.format(element)
        
        self.df = df
        self.dependent = dependent
        self.independent = independent
        self.result = ols(formula, df).fit()
    
    
    def summary(self):
        """
        Return linear regression summary
        """
        return self.result.summary()
    
    
    def plot_all(self):
        """
        Plot all dependent and independent variables against time. To visualize
        there relations
        """
        df = self.df
        independent = self.independent
        dependent = self.dependent
        
        plt.figure(figsize=(10, 5))
        plt.plot(df['date'], df[dependent], label=dependent)
        for indep in independent:
            plt.plot(df['date'], df[indep], label=indep)
        plt.xticks(rotation='vertical')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
        plt.show()
        
    
    def plot2D(self, rotation=False):
        """
        Print scatter plot and the best fit line
        Pre-condition:
            graph must be of 2D
        """
        if len(self.independent) > 1:
            raise ValueError("Not a single independent variable regression")
        params = self.result.params
        df = self.df
        k = params[1]
        b = params[0]
        independent = self.independent[0]
        dependent = self.dependent
        model = k*df[independent] + b
        
        plt.figure(figsize=(10, 5))
        plt.plot(df[independent], df[dependent], 'o')
        plt.plot(df[independent], model)
        plt.xlabel(independent)
        plt.ylabel(dependent)
        plt.title(dependent + ' vs. ' + independent)
        if rotation:
            plt.xticks(rotation='vertical')
        plt.show()


    def residual(self):
        """
        Return a pandas Series of residual
        Pre-condition:
            There should be no NAN in data. Hence length of date is equal to length
            of data
        """
        df = self.result.resid
        df.index = self.df['date']
        return df
    
    def residual_plot(self, std_line=2, rotation=True):
        """
        Plot the residual against time
        Pre-condition:
            std_line - plot n std band. Set to zero to disable the feature.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(self.df['date'], self.result.resid, label='residual')
        if rotation:
            plt.xticks(rotation='vertical')
        plt.title('residual plot')
        if std_line != 0:
            df = self.df
            std = self.residual().describe()['std']
            mean = self.residual().describe()['mean']
            num = len(df['date'])
            plt.plot(df['date'], std_line*std*np.ones(num)+mean, 'r--')
            plt.plot(df['date'], -std_line*std*np.ones(num)+mean, 'r--')
            plt.title('residual plot ({} STD band)'.format(std_line))
        plt.show()
        
    def residual_vs_fit(self):
        residual = self.residual()
        df = self.df
        y_predict = self.result.predict(df[self.independent])
        plt.plot(y_predict, residual, 'o')
        plt.plot(y_predict, np.zeros(len(residual)), 'r--')
        plt.xlabel("predict")
        plt.ylabel('residual')
        plt.title('Residual vs fit')
        plt.show()
      
    def train_test_sets(self):
        """
        test regression by seperating data to a train set and a test set
        """
        dependent = self.dependent
        independent = self.independent
        df = self.df
        y = df[dependent]
        X = df[independent]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
        # Instantiate model
        lm = LinearRegression()
        # Fit Model
        lm.fit(X_train, y_train)
        # Predict
        y_pred = lm.predict(X_test)
        
        RMSE = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
        print ("RMSE: %.5f" % RMSE)
        return RMSE
    
    def coefficient_consistency(self, segments=2):
        test_data = list()
        df = self.df
        for i in range(segments):
            test_data.append(df.iloc[i*len(df)/segments:(i+1)*len(df)/segments])
        for i in range(segments):
            reg = Regression(test_data[i], self.dependent, self.independent)
            string = []
            for j in range(len(reg.independent)):
                string.append("%.4f * %s" % (reg.result.params[reg.independent[j]], reg.independent[j]))
            print ("Period %d: %s = %s + %.4f" % (i, reg.dependent, ' + '.join(string), reg.result.params[0]))
        string = []
        for i in range(len(self.independent)):
            string.append("%.4f * %s" % (self.result.params[self.independent[i]], self.independent[i]))
        print ("Whole period: %s = %s + %.4f" % (self.dependent, ' + '.join(string), self.result.params[0]))
        
    def run_all(self):
        """
        Lazy ass's ultimate solution. Run all available analysis
        Pre-condition: 
            There should be only one independent variable
        """
        _2D = len(self.independent) == 1
        print 
        self.plot_all()
        print
        print self.summary()
        if _2D:
            self.plot2D()
        print
        print 'Error statistics'
        print self.residual().describe()
        print
        self.train_test_sets()
        self.residual_vs_fit()
        self.residual_plot()
        residual = self.residual()
        test_mean_reverting(residual)
        print
        print  'Halflife = ',half_life(residual)
        
        
    def summarize_all(self):
        if len(self.independent) == 1:
            dependent = self.dependent
            independent = self.independent[0]
            params = self.result.params
            result = self.result
            k = params[1]
            b = params[0]
            conf = result.conf_int()
            cadf = adfuller(result.resid)
            if cadf[0] <= cadf[4]['5%']:
                boolean = 'likely'
            else:
                boolean = 'unlikely'
            print
            print ("{:^40}".format("{} vs {}".format(dependent.upper(), independent.upper())))
            print ("%20s %s = %.4f * %s + %.4f" % ("Model:", dependent, k, independent, b))
            print ("%20s %.4f" % ("R square:",result.rsquared))
            print ("%20s [%.4f, %.4f]" % ("Confidence interval:", conf.iloc[1,0], conf.iloc[1,1]))
            print ("%20s %.4f" % ("Model error:", result.resid.std()))
            print ("%20s %s" % ("Mean reverting:", boolean))
            print ("%20s %d" % ("Half life:", half_life(result.resid)))
        else:
            dependent = self.dependent
            independent = self.independent      # list
            params = self.result.params
            result = self.result
            b = params[0]
            conf = result.conf_int()            # pandas 
            cadf = adfuller(result.resid)
            if cadf[0] <= cadf[4]['5%']:
                boolean = 'likely'
            else:
                boolean = 'unlikely'
            print
            print ("{:^40}".format("{} vs {}".format(dependent.upper(), (', '.join(independent)).upper())))
            string = []
            for i in range(len(independent)):
                string.append("%.4f * %s" % (params[independent[i]], independent[i]))
            print ("%20s %s = %s + %.4f" % ("Model:", dependent, ' + '.join(string), b))
            print ("%20s %.4f" % ("R square:",result.rsquared))
            string = []
            for i in range(len(independent)):
                string.append("[%.4f, %.4f]" % (conf.loc[independent[i], 0], conf.loc[independent[i], 1]))
            print ("%20s %s" % ("Confidence interval:", ' , '.join(string)))
            print ("%20s %.4f" % ("Model error:", result.resid.std()))
            print ("%20s %s" % ("Mean reverting:", boolean))
            print ("%20s %d" % ("Half life:", half_life(result.resid)))
 


       
###############################################################################
"""                     Kalman Filter in Regression                         """
###############################################################################


class KalmanRegression(object):
    def __init__(self, df, dependent=None, independent=None, delta=None, trans_cov=None, obs_cov=None):
        if not dependent:
            dependent = df.columns[1]
        if not independent:
            independent = df.columns[2]
          
        self.x = df[independent]
        self.x.index = df['date']
        self.y = df[dependent]
        self.y.index = df['date']
        self.dependent = dependent
        self.independent = independent
        
        self.delta = delta or 1e-5
        self.trans_cov = trans_cov or self.delta / (1 - self.delta) * np.eye(2)
        self.obs_mat = np.expand_dims(
                np.vstack([[self.x.values], [np.ones(len(self.x))]]).T,
                axis = 1
                )
        self.obs_cov = obs_cov or 1
        self.kf = KalmanFilter(n_dim_obs=1, n_dim_state=2,
                               initial_state_mean=np.zeros(2),
                               initial_state_covariance=np.ones((2, 2)),
                               transition_matrices=np.eye(2),
                               observation_matrices=self.obs_mat,
                               observation_covariance=self.obs_cov,
                               transition_covariance=self.trans_cov)
        self.state_means, self.state_covs = self.kf.filter(self.y.values)
        
    def slope(self):
        state_means = self.state_means
        return pd.Series(state_means[:,0], index=self.x.index)
    
    def plot_params(self):
        state_means = self.state_means
        x = self.x
        _, axarr = plt.subplots(2, sharex=True)
        axarr[0].plot(x.index, state_means[:,0], label='slope')
        axarr[0].legend()
        axarr[1].plot(x.index, state_means[:,1], label='intercept')
        axarr[1].legend()
        plt.tight_layout()
        plt.show()
        return state_means[:,0]
        
    def plot2D(self):
        x = self.x
        y = self.y
        state_means = self.state_means
        
        cm = plt.get_cmap('jet')
        colors = np.linspace(0.1, 1, len(x))
        # Plot data points using colormap
        sc = plt.scatter(x, y, s=30, c=colors, cmap=cm, edgecolor='k', alpha=0.7)
        cb = plt.colorbar(sc)
        cb.ax.set_yticklabels([str(p.date()) for p in x[::len(x)//9].index])
        
        # Plot every fifth line
        step = 100
        xi = np.linspace(x.min()-5, x.max()+5, 2)
        colors_l = np.linspace(0.1, 1, len(state_means[::step]))
        for i, beta in enumerate(state_means[::step]):
            plt.plot(xi, beta[0] * xi + beta[1], alpha=.2, lw=1, c=cm(colors_l[i]))
            
        # Plot the OLS regression line
        plt.plot(xi, poly1d(np.polyfit(x, y, 1))(xi), '0.4')
        
        plt.title(self.dependent + ' vs. ' + self.independent)
        plt.show()
        
    def run_all(self):
        self.plot_params()
        self.plot2D()